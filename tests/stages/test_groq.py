# from harness.models import TaskRequest
# from harness.stages.research import research

# task = TaskRequest(
#     task=(
#         "Design a small Python library that validates "
#         "structured configuration files using Pydantic."
#     ),
#     parameters={
#         "requirements": [
#             "Support JSON configuration",
#             "Provide descriptive validation errors",
#             "Include unit tests",
#         ]
#     },
# )

# result = research({"task": task})

# print(result["research_result"].model_dump_json(indent=2))