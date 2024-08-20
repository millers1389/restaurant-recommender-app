# import os
# import arize-otel

# # Import open-telemetry dependencies
# from arize_otel import register_otel, Endpoints

# SPACE_ID = "SPACE_ID" # Change this line
# API_KEY = "API_KEY" # Change this line

# model_id = "tutorial-otlp-tracing-langchain-rag"
# model_version = "1.0"

# if SPACE_ID == "SPACE_ID" or API_KEY == "API_KEY":
#     raise ValueError("❌ CHANGE SPACE_ID AND/OR API_KEY")
# else:
#     print("✅ Import and Setup Arize Client Done! Now we can start using Arize!")

# # Setup OTEL via our convenience function
# register_otel(
#     endpoints = Endpoints.ARIZE,
#     space_id = "your-space-id", # in app Space settings page
#     api_key = "your-api-key", # in app Space settings page
#     model_id = "your-model-id",
# )
# # Import the automatic instrumentor from OpenInference
# from openinference.instrumentation.langchain import LangChainInstrumentor

# # Finish automatic instrumentation
# LangChainInstrumentor().instrument()

import os
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry import trace as trace_api
from openinference.instrumentation.langchain import LangChainInstrumentor
# import arize-otel


def arize_instrument(space_id, api_key, model_id, model_version):
    # Set the Space and API keys as headers
    os.environ['OTEL_EXPORTER_OTLP_TRACES_HEADERS'] = f"space_id={space_id},api_key={api_key}"

    # Set the model id and version as resource attributes
    resource = Resource(
        attributes={
            "model_id": model_id,
            "model_version": model_version,
        }
    )

    endpoint = "https://otlp.arize.com/v1"
    span_exporter = OTLPSpanExporter(endpoint=endpoint)
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor=span_processor)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)

    # Instrument LangChain
    LangChainInstrumentor().instrument()

    print("Arize setup completed successfully.")

if __name__ == "__main__":
    # This allows you to run the setup directly if needed
    arize_instrument("YOUR_SPACE_ID", "YOUR_API_KEY", "your-model-id", "your-model-version")