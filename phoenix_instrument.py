import os
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.langchain import LangChainInstrumentor

def setup_phoenix_instrumentation(endpoint="http://localhost:6006/v1/traces", model_id="raleigh-restaurant-recommender", model_version="v1.0"):
    resource = Resource(attributes={
        "service.name": "raleigh-restaurant-recommender",
        "model_id": model_id,
        "model_version": model_version
    })
    
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=endpoint)
    span_processor = SimpleSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    trace_api.set_tracer_provider(tracer_provider)

    # Instrument LangChain
    LangChainInstrumentor().instrument()

    print("Phoenix instrumentation set up successfully.")

if __name__ == "__main__":
    setup_phoenix_instrumentation()
