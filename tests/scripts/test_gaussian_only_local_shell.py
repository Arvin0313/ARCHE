import os
import sys
import json
import time
from typing import Any, Dict

PROJECT_ROOT = "/home/lidong/ChemistryAgent"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from chemistry_multiagent.agents.execution_agent import ExecutionAgent, ExecutionStep

TOOLPOOL = os.path.join(
    PROJECT_ROOT, "src", "chemistry_multiagent", "tools", "toolpool.json"
)
WORK_DIR = os.path.join(PROJECT_ROOT, "tests", "temp", "gaussian_only_local")
os.makedirs(WORK_DIR, exist_ok=True)

GJF_PATH = os.path.join(PROJECT_ROOT, "tests", "inputs", "gjf", "water.gjf")


def dump_json(name: str, obj: Dict[str, Any]) -> None:
    path = os.path.join(WORK_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"saved: {path}")


def summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": result.get("success"),
        "execution_mode": result.get("execution_mode"),
        "scheduler": result.get("scheduler"),
        "status": result.get("status"),
        "submitted": result.get("submitted"),
        "completed": result.get("completed"),
        "job_id": result.get("job_id"),
        "work_dir": result.get("work_dir"),
        "log_path": (result.get("output_artifacts") or {}).get("log"),
        "chk_path": (result.get("output_artifacts") or {}).get("chk"),
        "message": result.get("message"),
        "error": result.get("error"),
        "has_parsed_results": isinstance(result.get("parsed_results"), dict),
    }


def main():
    if not os.path.exists(GJF_PATH):
        raise FileNotFoundError(f"GJF file not found: {GJF_PATH}")

    agent = ExecutionAgent(
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
        toolpool_path=TOOLPOOL,
        expert_model_name=os.environ.get("ARCHE_CHEM_MODEL_NAME", "qwen2.5-7b-instruct"),
        expert_model_path=os.environ.get("ARCHE_CHEM_MODEL_PATH"),
        expert_backend=os.environ.get("ARCHE_CHEM_BACKEND", "local_hf"),
        enable_expert_analysis=True,
        gaussian_execution_mode="local_shell",
        gaussian_command=os.environ.get("GAUSSIAN_COMMAND", "g16"),
        gaussian_job_root=WORK_DIR,
    )

    input_data = {
        "gjf_path": GJF_PATH,
        "input_path": GJF_PATH,
        "work_dir": WORK_DIR,
        "job_name": "water_opt_test",
        "job_type": "opt",
        "route_section": "#p HF/3-21G Opt",
    }

    step = ExecutionStep(
        step_number=1,
        description="Run Gaussian optimization for water molecule",
        tool_name="Gaussian",
        expected_input=GJF_PATH,
        expected_output=os.path.join(WORK_DIR, "water.log"),
        input_data=input_data,
        output_files=[],
        job_type="opt",
        route_section="#p HF/3-21G Opt",
        scientific_context={
            "scientific_question": "Test Gaussian local_shell backend with a minimal water optimization",
            "chemistry_context": {
                "candidate_elements": ["H", "O"],
                "needs_ts": False,
                "needs_irc": False
            }
        },
        working_directory=WORK_DIR,
        artifacts=[
            {
                "type": "input_file",
                "path": GJF_PATH,
                "format": "gjf",
            }
        ],
    )

    print("=== first execution ===")
    result1 = agent.execute_gaussian_related_tool(
        tool_name="Gaussian",
        input_data=input_data,
        step=step,
    )
    dump_json("result_first.json", result1)

    summary1 = summarize_result(result1)
    dump_json("summary_first.json", summary1)
    print(json.dumps(summary1, indent=2, ensure_ascii=False))

    print("=== wait 5 seconds ===")
    time.sleep(5)

    print("=== second execution / recovery ===")
    result2 = agent.execute_gaussian_related_tool(
        tool_name="Gaussian",
        input_data=input_data,
        step=step,
    )
    dump_json("result_second.json", result2)

    summary2 = summarize_result(result2)
    dump_json("summary_second.json", summary2)
    print(json.dumps(summary2, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()