# Deploy Agata on AWS

Use one private S3 bucket for daily editions, an ECS Fargate task for generation, and API Gateway with Lambda for REST and the Telegram webhook. A manual generation run is enough to validate the stack before enabling a schedule.

## Build the images

Create two private ECR repositories in the deployment region. Set `AWS_REGION`, `ECR_REGISTRY` (the registry hostname) and `IMAGE_TAG` (for example, a Git commit SHA). Match the image platform to the runtime architecture. The example uses x86-64.

```sh
aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$ECR_REGISTRY"

docker buildx build --platform linux/amd64 --provenance=false --target app \
  -t "$ECR_REGISTRY/agata-job:$IMAGE_TAG" --push .
docker buildx build --platform linux/amd64 --provenance=false --target lambda \
  -t "$ECR_REGISTRY/agata-api:$IMAGE_TAG" --push .
```

Use the resulting digests in ECS and Lambda. [Lambda container image requirements](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)

## Bucket and generation task

1. Create a private S3 bucket and enable versioning to preserve the previous edition after recovery. Both runtimes need `s3:ListBucket` on this bucket and `s3:GetObject` on `daily/*`. The generation task also needs `s3:PutObject` on `daily/*`. Grant permissions through runtime roles. The read path uses bucket permission to have S3 return a missing-key response for unpublished days.
2. Register a Fargate task definition using the job image and `awsvpc` networking. Use an ECS execution role for ECR pulls and CloudWatch logs. Set `AGATA_S3_BUCKET`, `AGATA_TIMEZONE`, and `AGATA_RELEASE_ID`. Configure a network route to ECR and S3. Override the image command with `["python", "-m", "agata", "generate"]` for a test run.
3. Check the task exit code and logs, and verify that `daily/YYYY-MM-DD.json` contains 48 forecasts. Run the same command again to verify that it reports `already_published`.

For gRPC inference, configure the ECS task with `AGATA_GENERATOR=grpc`, `AGATA_MODEL_GRPC_TARGET`, and `AGATA_MODEL_ID`. Give the task network access to the model server. Inject an optional bearer token from Secrets Manager. The REST and Telegram Lambda still reads completed editions from S3.

[S3 missing-key behavior](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html)

## Schedule the batch from the planned timestamp

Create a Standard Step Functions state machine with the following definition. Replace the ARNs, container name, and subnet and security group IDs with your ECS configuration. The example assumes one job container. Private subnets need the network access described above.

```json
{
  "StartAt": "Generate",
  "States": {
    "Generate": {
      "Type": "Task",
      "Resource": "arn:aws:states:::ecs:runTask.sync",
      "Parameters": {
        "Cluster": "CLUSTER_ARN",
        "TaskDefinition": "TASK_DEFINITION_ARN",
        "LaunchType": "FARGATE",
        "NetworkConfiguration": {
          "AwsvpcConfiguration": {
            "Subnets": ["SUBNET_ID"],
            "SecurityGroups": ["SECURITY_GROUP_ID"],
            "AssignPublicIp": "DISABLED"
          }
        },
        "Overrides": {
          "ContainerOverrides": [{
            "Name": "JOB_CONTAINER_NAME",
            "Command.$": "States.Array('python', '-m', 'agata', 'generate', '--scheduled-time', $.scheduled_time)"
          }]
        }
      },
      "Next": "Check exit code"
    },
    "Check exit code": {
      "Type": "Choice",
      "Choices": [{
        "And": [
          {"Variable": "$.Containers[0].ExitCode", "IsPresent": true},
          {"Variable": "$.Containers[0].ExitCode", "NumericEquals": 0}
        ],
        "Next": "Done"
      }],
      "Default": "Generation failed"
    },
    "Done": {"Type": "Succeed"},
    "Generation failed": {"Type": "Fail", "Error": "GenerationFailed"}
  }
}
```

Give the state machine role permissions for the ECS sync integration and `iam:PassRole` for the ECS task and execution roles. In EventBridge Scheduler, use `cron(0 23 * * ? *)`, set the schedule timezone to `Europe/Moscow`, disable the flexible window, and select the state machine as the target. Give the scheduler role `states:StartExecution` on that machine and use the following target input.

```json
{"scheduled_time": "<aws.scheduler.scheduled-time>"}
```

Before enabling the schedule, start the state machine with a previous timestamp and confirm the task log shows the expected `scheduled_time` and edition `date`. For example, `2026-12-01T20:00:00Z` targets December 2 in Moscow even when the container starts after midnight. To replay a missing edition manually, run the task with `generate --date YYYY-MM-DD`.

[EventBridge schedule context](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule-context-attributes.html) · [Step Functions ECS integration and IAM](https://docs.aws.amazon.com/step-functions/latest/dg/connect-ecs.html)

## Recover a fallback edition

In the job's CloudWatch log group, create a metric filter with pattern `{ $.status = "published" && $.health = "degraded" }` and metric value `1`. An alarm on that metric can notify the operator even when ECS exits successfully. A count of `fallbacks: 48` indicates a full fallback edition.

When the model is healthy, run the generation task with the model generator configured and override the command to `["python", "-m", "agata", "recover", "--date", "YYYY-MM-DD"]`. Check the output for `recovered` and the remaining fallback count. If the write reports `conflict`, inspect the published edition and retry if fallback forecasts remain. Bucket versioning retains the previous object. For rollback, identify the version in S3 and restore it as the current version while coordinating any recovery tasks for that date.

## Lambda API and Telegram

1. Create a Lambda function from the API image with a 15-second timeout and an execution role for read-only edition access and CloudWatch logging. Set the same `AGATA_S3_BUCKET` and `AGATA_TIMEZONE` as the job. Attach an API Gateway HTTP API Lambda proxy integration with payload format 2.0. The default stage exposes `/healthz`, `/readyz`, `/docs`, and the two versioned routes.
2. In Secrets Manager, store a JSON object with keys `bot_token` and `webhook_secret`. Use a separate random URL-safe value for the webhook secret. Give the Lambda execution role permission to read that secret and set `AGATA_TELEGRAM_SECRET_ARN` to its ARN.
3. Register the webhook from a checkout configured with the same secret. The CLI reads it from Secrets Manager when `AGATA_TELEGRAM_SECRET_ARN` is present and AWS credentials are available.

```sh
python -m agata set-webhook 'https://your-api.execute-api.example/v1/telegram/webhook'
```

4. Check `/readyz` and the REST forecast endpoint, then send `/start` to the bot. Run the batch with a distinct date when testing a new model release. To roll back generation for future dates, switch the ECS schedule to the previous task definition.

[API Gateway Lambda proxy integration](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html) · [Telegram webhook setup](https://core.telegram.org/bots/api#setwebhook)
