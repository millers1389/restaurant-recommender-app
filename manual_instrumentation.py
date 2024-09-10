import os
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from openinference.semconv.trace import SpanAttributes

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

    print("Arize setup completed successfully.")

def create_custom_span(tracer, span_name: str, attributes: dict):
    with tracer.start_as_current_span(span_name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        # This is where you would add the code that you want to trace manually
        print(f"Span {span_name} created with attributes: {attributes}")
        return span

def trace_function_execution(tracer, function, *args, **kwargs):
    span_name = f"Execution of {function.__name__}"
    attributes = {"function_name": function.__name__}

    with tracer.start_as_current_span(span_name) as span:
        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        # Execute the function and capture its output
        result = function(*args, **kwargs)
        
        # Optionally, you can add more attributes or events after execution
        span.set_attribute("function_result", str(result))
        print(f"Function {function.__name__} executed and traced with result: {result}")
        
        return result

# Example functions to trace
def example_function_1(param1, param2):
    # Simulate some processing
    result = param1 + param2
    return result

def example_function_2(param):
    # Simulate some processing
    result = param * 2
    return result

if __name__ == "__main__":
    # This allows you to run the setup directly if needed
    arize_instrument("YOUR_SPACE_ID", "YOUR_API_KEY", "your-model-id", "your-model-version")
    
    tracer = trace_api.get_tracer(__name__)

    # Manually create custom spans
    create_custom_span(tracer, "custom-span-1", {"attribute_key_1": "value_1"})
    create_custom_span(tracer, "custom-span-2", {"attribute_key_2": "value_2"})

    # Trace the execution of specific functions
    result1 = trace_function_execution(tracer, example_function_1, 5, 10)
    result2 = trace_function_execution(tracer, example_function_2, 7)


