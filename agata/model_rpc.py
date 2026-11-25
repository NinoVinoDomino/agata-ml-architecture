from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import grpc
import httpx

from agata.config import Settings
from agata.domain import Sign, Topic
from agata.generation import (
    GeneratedText,
    GenerationRequest,
    GenerationUnavailable,
    Generator,
    OpenAICompatibleGenerator,
    TemplateGenerator,
    format_prompt,
)
from agata.rpc import model_pb2, model_pb2_grpc


class GrpcGenerator:
    def __init__(
        self,
        channel: grpc.aio.Channel,
        model_id: str,
        timeout: float,
        api_key: str | None = None,
    ):
        self.id = model_id
        self.stub = model_pb2_grpc.ModelStub(channel)
        self.timeout = timeout
        self.metadata = (("authorization", f"Bearer {api_key}"),) if api_key else ()

    async def generate(self, request: GenerationRequest) -> GeneratedText:
        system, user = format_prompt(request)
        try:
            reply = await self.stub.Generate(
                model_pb2.GenerateRequest(
                    model_id=self.id,
                    date=str(request.date),
                    sign=request.sign,
                    topic=request.topic,
                    attempt=request.attempt,
                    system_prompt=system,
                    user_prompt=user,
                ),
                timeout=self.timeout,
                metadata=self.metadata,
            )
        except grpc.aio.AioRpcError as exc:
            raise GenerationUnavailable(f"ошибка gRPC: {exc.code().name}") from None
        source = "template" if reply.editorial_template else "model"
        return GeneratedText(reply.text.strip(), source)


@asynccontextmanager
async def open_generator(settings: Settings) -> AsyncIterator[Generator]:
    if settings.generator == "template":
        yield TemplateGenerator()
        return
    key = settings.model_api_key.get_secret_value() if settings.model_api_key else None
    model_id = settings.model_id
    assert model_id is not None
    if settings.generator == "grpc":
        target = settings.model_grpc_target
        assert target is not None
        if settings.model_grpc_tls:
            channel = grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
        else:
            channel = grpc.aio.insecure_channel(target)
        async with channel:
            yield GrpcGenerator(channel, model_id, settings.model_timeout, key)
        return
    endpoint = settings.model_endpoint
    assert endpoint is not None
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=settings.model_timeout, headers=headers) as client:
        yield OpenAICompatibleGenerator(client, str(endpoint), model_id)


class TemplateModelService(model_pb2_grpc.ModelServicer):
    """A local gRPC example using the same editorial templates as the batch job."""

    async def Generate(self, request, context):
        try:
            generation = GenerationRequest(
                date.fromisoformat(request.date),
                Sign(request.sign),
                Topic(request.topic),
                request.attempt,
            )
            if request.model_id != TemplateGenerator.id or request.attempt not in (1, 2):
                raise ValueError
        except ValueError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid template model request")
        output = await TemplateGenerator().generate(generation)
        return model_pb2.GenerateReply(text=output.text, editorial_template=True)


async def serve_template_model(port: int) -> None:
    server = grpc.aio.server()
    model_pb2_grpc.add_ModelServicer_to_server(TemplateModelService(), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    print(f"Template model listening on 127.0.0.1:{port}", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=1)
