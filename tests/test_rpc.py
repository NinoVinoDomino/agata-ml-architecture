from datetime import date

import grpc
import pytest

from agata.config import Settings
from agata.domain import Sign, Topic
from agata.generation import ForecastWorkflow
from agata.model_rpc import TemplateModelService, open_generator
from agata.rpc import model_pb2_grpc


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_forecast_workflow_over_grpc():
    server = grpc.aio.server()
    model_pb2_grpc.add_ModelServicer_to_server(TemplateModelService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        base = {
            "generator": "grpc",
            "model_grpc_target": f"127.0.0.1:{port}",
            "model_grpc_tls": False,
            "model_timeout": 2,
        }
        day = date(2026, 1, 1)
        async with open_generator(Settings(**base, model_id="editorial-v1")) as generator:
            forecast = await ForecastWorkflow(generator).run(day, Sign.LEO, Topic.FINANCE)
            assert forecast.source == "template"
            assert forecast.attempts == 1
            assert forecast.text.startswith("Лев · Финансы.")

        async with open_generator(Settings(**base, model_id="missing")) as generator:
            forecast = await ForecastWorkflow(generator).run(day, Sign.LEO, Topic.FINANCE)
            assert forecast.source == "fallback"
            assert forecast.attempts == 2
            assert forecast.validation_issue == "ошибка gRPC: INVALID_ARGUMENT"
    finally:
        await server.stop(grace=0)
