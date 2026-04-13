# Includes/setup_orchestrator.py

from pathlib import Path
from typing import Any, Dict, Optional

from .catalog_utils import build_user_catalog, _current_user_email, _vocareum_schema_name
from .volume_utils import create_volume
from .process_data import process_csv_data, create_sample_labels
from .pdf_creation import create_listings_pdf
from .genie_creation import create_genie_space
from .genie_deletion import delete_genie_space_by_title
from .config_loader import load_config, ConfigLoader
from .manifest_validator import validate_manifest_alignment
from .compute_check import _serverless_version_check
from .artifacts_manager import ArtifactsManager
from .experiment_manager import ExperimentManager
from .config_renderer import ConfigRenderer
from .tool_manager import ToolManager
from .agent_manager import AgentManager
from .create_files_folders import create_file, create_folder

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_share_name(user_email: str, share_name: str, vocareum_share_name: str) -> str:
    """Return the correct Marketplace share name for Vocareum vs other workspaces."""
    if user_email.lower().endswith("@vocareum.com"):
        print(f"  Using Vocareum Delta Share: {vocareum_share_name}")
        return "dbacademy_" + vocareum_share_name
    print(f"  Using Marketplace Delta Share: {share_name}")
    return share_name

def _set_lakebase_project_name(user_email: str) -> str:
    """Return the Lakebase project name derived from the user's email.

    'first.last@email.com' → 'agent-memory-first-last'
    """
    local_part = user_email.lower().split("@")[0]
    cleaned = local_part.replace(".", "-").replace("_", "-")
    return f"agent-memory-{cleaned}"


bool_params = {"dev_mode", "resource_cleanup", "deploy"}

config_mapping = {
    # ---- Environment ----
    "course_name":                      "course_name",
    "catalog_name":                     "catalog.name",
    "catalog_prefix":                   "catalog.prefix",
    "schema_name":                      "catalog.schema_name",
    "volume_name":                      "catalog.volume_name",
    "databricks_share_name":            "data.databricks_share_name",
    "vocareum_share_name":              "data.vocareum_share_name",
    "table_name":                       "data.table_name",
    "warehouse_name":                   "genie.warehouse_name",
    "genie_space_title":                "genie.space_title",
    "genie_space_description":          "genie.space_description",
    "genie_source_table":               "genie.source_table",
    "dev_mode":                         "dev_mode",
    "medallion_lab":                    "medallion_lab",
    "serverless_compute_version":       "serverless_compute_version",
    "resource_cleanup":                 "resource_cleanup",
    # ---- Agents ----
    "llm_endpoint_name":                "agents.llm_endpoint_name",
    "alias":                            "agents.alias",
    "correctness_eval_endpoint":        "agents.eval_endpoints.correctness",
    "retrieval_sufficiency_endpoint":   "agents.eval_endpoints.retrieval_sufficiency",
    "custom_eval_endpoint":             "agents.eval_endpoints.custom",
    "guidelines_endpoint":              "agents.eval_endpoints.guidelines",
    "recall_eval_endpoint":             "agents.eval_endpoints.recall",
    "deploy":                           "agents.deploy",
    "lakebase_instance_name":           "agents.lakebase_instance_name",
    "lakebase_autoscaling_branch":      "agents.lakebase_autoscaling_branch",
    # ---- Apps (Databricks Apps / DAB deployment) ----
    "app_serving_endpoint_name":        "apps.serving_endpoint_name",
    "app_bundle_name":                  "apps.bundle_name",
    # ---- File Creation ----
    "folder":                           "files.file1.folder_name",
    "file1":                            "files.file1.name",
    "file1_contents":                   "files.file1.file1_contents",
    "file2":                            "files.file2.name",
    "file2_contents":                   "files.file2.file2_contents",
    "file3":                            "files.file3.name",
    "file3_contents":                   "files.file3.file3_contents",
    "file4":                            "files.file4.name",
    "file4_contents":                   "files.file4.file4_contents",
    "file5":                            "files.file5.name",
    "file5_contents":                   "files.file5.file5_contents",
    "file6":                            "files.file6.name",
    "file6_contents":                   "files.file6.file6_contents",
    "file7":                            "files.file7.name",
    "file7_contents":                   "files.file7.file7_contents",
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _merge_params(
    cfg: ConfigLoader,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge config values with runtime overrides (overrides take precedence)."""
    defaults = {
        "catalog_prefix": "labuser",
        "alias":          "champion",
        "deploy":         False,
    }

    result = {}
    for param, config_path in config_mapping.items():
        if overrides.get(param) is not None:
            result[param] = overrides[param]
        else:
            config_value = cfg.get(config_path)
            if config_value is not None:
                result[param] = config_value
            else:
                result[param] = defaults.get(param)

    for param in bool_params:
        if result.get(param) is not None:
            result[param] = _parse_bool(result[param])

    return result


def _register_python_uc_tools(tool_names: list, catalog_name: str, schema_name: str) -> None:
    """Load Python tool files from Includes_v2/tools/python/ and register as UC functions."""
    import importlib.util
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient

    tools_dir = Path(__file__).parent.parent / "tools" / "python"
    client = DatabricksFunctionClient(execution_mode="serverless")

    for tool_name in tool_names:
        tool_path = tools_dir / f"{tool_name}.py"
        if not tool_path.exists():
            print(f"  Warning: Python tool not found: {tool_path}")
            continue

        spec = importlib.util.spec_from_file_location(tool_name, tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, tool_name)

        client.create_python_function(
            func=func,
            catalog=catalog_name,
            schema=schema_name,
            replace=True,
        )
        print(f"  Registered Python UC function: {tool_name}")

def dev_classroom_cleanup(
    dev_mode: bool,
    resource_cleanup: bool,
    catalog_name: str,
    step_index: int,
    artifacts_dir: Optional[Path] = None,
    created_folders: list = None,
    experiment_names: list = None,
    genie_space_title: str = None,
    user_email: str = None,
):
    print("Refreshing environment")
    if not (dev_mode and resource_cleanup):
        return

    import shutil
    import mlflow
    from mlflow import MlflowClient

    print(f"\n[{step_index}] Cleaning up resources...")

    # 1. Drop UC catalog — removes all schemas, tables, volumes, UC functions, registered models
    spark.sql(f"DROP CATALOG IF EXISTS {catalog_name} CASCADE")
    print(f"  Catalog dropped: {catalog_name}")

    # 2. Delete local artifacts folder
    if artifacts_dir and artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
        print(f"  Artifacts folder deleted: {artifacts_dir}")

    # 3. Delete folders created by the files: config section
    for folder in (created_folders or []):
        folder = Path(folder)
        if folder.exists():
            shutil.rmtree(folder)
            print(f"  Folder deleted: {folder}")

    # 4. Delete MLflow experiments created during agent setup
    if experiment_names and user_email:
        mlflow.set_tracking_uri("databricks")
        client = MlflowClient()
        for exp_name in experiment_names:
            exp_path = f"/Workspace/Users/{user_email}/{exp_name}"
            exp = mlflow.get_experiment_by_name(exp_path)
            if exp:
                client.delete_experiment(exp.experiment_id)
                print(f"  MLflow experiment deleted: {exp_name}")

    # 5. Delete Genie space (silently skips if not found)
    if genie_space_title:
        try:
            delete_genie_space_by_title(genie_space_title)
        except Exception as e:
            print(f"  Warning: Could not delete Genie space '{genie_space_title}': {e}")

    print("  All resources cleaned up. Set resource_cleanup: false to disable.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def setup_demo_environment(
    config_path: Optional[str | Path] = None,
    agents_dir: Optional[str | Path] = None,
    agent_configs_dir: Optional[str | Path] = None,
    agent_tools_dir: Optional[str | Path] = None,
    eval_datasets_dir: Optional[str | Path] = None,
    **overrides,
) -> Dict[str, Any]:
    """
    Orchestrate the full demo environment setup.

    Runs environment steps driven by the config YAML (catalog, schema, volume,
    UC tools, data, PDF, Genie) and, when ``agents.llm_endpoint_name`` is
    configured, also runs agent steps (artifact copy, tool creation, MLflow
    registration, endpoint deployment).

    Parameters
    ----------
    config_path : str | Path, optional
        Path to a config YAML file.
    agents_dir : str | Path, optional
        Source directory containing agent .py files.
        Defaults to <includes_folder>/agents relative to the config file.
    agent_configs_dir : str | Path, optional
        Source directory containing agent YAML config templates.
        Defaults to <includes_folder>/agent configs relative to the config file.
    agent_tools_dir : str | Path, optional
        Base directory containing per-agent tool subfolders (each subfolder
        holds .txt DDL files for that agent's UC functions).
        Defaults to <includes_folder>/agent tools relative to the config file.
    eval_datasets_dir : str | Path, optional
        Source directory containing evaluation dataset JSON files.
        Defaults to <includes_folder>/evaluation_datasets relative to the config file.
    **overrides
        Override any config value at runtime. Keys match ``config_mapping``
        (e.g. ``schema_name``, ``llm_endpoint_name``).

    Returns
    -------
    dict
        catalog_name, schema_name, volume_path, username, table_name,
        pdf_path, genie_space_id, agent_configs, deployed_endpoint
    """
    cfg = load_config(config_path)
    cfg.validate()
    params = _merge_params(cfg, overrides)

    # Resolve resource dirs relative to the includes folder (config's grandparent dir).
    # Using .resolve() to get an absolute path so this works regardless of cwd.
    # Explicit overrides are always respected as-is.
    _includes_dir = Path(cfg.config_path).resolve().parent.parent
    agents_dir        = Path(agents_dir).resolve()        if agents_dir        else _includes_dir / "agents"
    agent_configs_dir = Path(agent_configs_dir).resolve() if agent_configs_dir else _includes_dir / "agent configs"
    agent_tools_dir   = Path(agent_tools_dir).resolve()   if agent_tools_dir   else _includes_dir / "agent tools"
    eval_datasets_dir = Path(eval_datasets_dir).resolve() if eval_datasets_dir else _includes_dir / "evaluation_datasets"

    print("=" * 60)
    print("Starting Demo Environment Setup")
    print(f"Using config: {cfg.config_path}")
    print("=" * 60)

    i = 0

    # ------------------------------------------------------------------
    # [i] Manifest validation (dev_mode only)
    # ------------------------------------------------------------------
    dev_mode = params.get("dev_mode")
    if dev_mode:
        print(f"\n[{i}] Validating manifest alignment...")
        validate_manifest_alignment(
            course_name=params.get("course_name"),
            vocareum_share_name=params.get("vocareum_share_name"),
            medallion_labs=params.get("medallion_lab"),
        )
        i += 1

    # ------------------------------------------------------------------
    # [i] Compute version check
    # ------------------------------------------------------------------
    serverless_version = params.get("serverless_compute_version")
    if serverless_version:
        if not _serverless_version_check(serverless_version):
            raise EnvironmentError(
                f"ERROR: This notebook must be run on Databricks Serverless "
                f"{serverless_version}. "
                "Please switch your compute and retry."
            )
        print(f"\n[{i}] Compute check passed: Serverless {serverless_version} detected.")
    i += 1

    # ------------------------------------------------------------------
    # [i] Resource cleanup (dev_mode + resource_cleanup only)
    # Resolve the full catalog name FIRST so we drop the right catalog,
    # then recreate it below.
    # ------------------------------------------------------------------
    resource_cleanup = params.get("resource_cleanup")

    if dev_mode:
        catalog_name_to_delete = params.get("catalog_name")
        if catalog_name_to_delete:
            catalog_name_to_delete = build_user_catalog(catalog_forced=catalog_name_to_delete)
        else:
            catalog_name_to_delete = build_user_catalog(prefix=params.get("catalog_prefix", "labuser"))
        try:
            dev_classroom_cleanup(dev_mode, resource_cleanup, catalog_name_to_delete, i)
        except Exception as e:
            print(f"ERROR during cleanup: {e}")

    # ------------------------------------------------------------------
    # [i] Catalog
    # ------------------------------------------------------------------
    print(f"\n[{i}] Setting up catalog...")
    catalog_name = params.get("catalog_name")
    if catalog_name:
        catalog_name = build_user_catalog(catalog_forced=catalog_name)
    else:
        catalog_name = build_user_catalog(prefix=params.get("catalog_prefix", "labuser"))
    user_email = _current_user_email()
    i += 1

    # ------------------------------------------------------------------
    # [i] Schema
    # ------------------------------------------------------------------
    # On Vocareum, each user's schema is their username (pre-provisioned).
    # Use it unless the caller explicitly passed a schema_name override.
    if user_email.lower().endswith("@vocareum.com") and "schema_name" not in overrides:
        schema_name = _vocareum_schema_name(user_email)
    else:
        schema_name = params.get("schema_name")
    if schema_name:
        print(f"\n[{i}] Setting up schema...")
        spark.sql(f"USE CATALOG {catalog_name}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        spark.sql(f"USE SCHEMA {schema_name}")
        print(f"  Catalog : {catalog_name}")
        print(f"  Schema  : {schema_name}")
    i += 1

    # Instantiate ToolManager here so it's available for both the standalone
    # tools step and (if running) the agent setup block.
    tool_manager = ToolManager(catalog_name=catalog_name, schema_name=schema_name)

    # ------------------------------------------------------------------
    # [i] Volume
    # ------------------------------------------------------------------
    volume_path = None
    volume_name = params.get("volume_name")
    if volume_name:
        print(f"\n[{i}] Creating UC volume...")
        volume_path = create_volume(catalog_name, schema_name, volume_name)
        i += 1

    # ------------------------------------------------------------------
    # [i] Folder + Files (optional)
    # ------------------------------------------------------------------
    file_folder_mapping = {
        k: v for k, v in config_mapping.items()
        if k == "folder" or k.startswith("file")
    }
    file_numbers = sorted(set(
        k.replace("file", "").replace("_contents", "")
        for k in file_folder_mapping
        if k.startswith("file")
        if k.replace("file", "").replace("_contents", "").isdigit()
    ))

    # Apply apps-specific substitutions to file contents before writing
    apps_cfg = cfg.get_apps_config()
    if apps_cfg:
        app_substitutions = {
            k: v for k, v in {
                "$APP_SERVING_ENDPOINT": params.get("app_serving_endpoint_name"),
                "$APP_BUNDLE_NAME":      params.get("app_bundle_name"),
            }.items() if v
        }
        if app_substitutions:
            for n in file_numbers:
                contents_key = f"file{n}_contents"
                if params.get(contents_key):
                    for placeholder, value in app_substitutions.items():
                        params[contents_key] = params[contents_key].replace(placeholder, value)

    if params.get("folder"):
        print(f"\n[{i}] Creating folder...")
        create_folder(f"./{params.get('folder')}")
        i += 1

    if any(params.get(f"file{n}") for n in file_numbers):
        print(f"\n[{i}] Creating files...")
        for n in file_numbers:
            filename = params.get(f"file{n}")
            contents = params.get(f"file{n}_contents")
            if filename:
                create_file(f"./{params.get('folder')}/{filename}", contents or "")
        i += 1


    # ------------------------------------------------------------------
    # [i] Data
    # ------------------------------------------------------------------
    data_config = cfg.get_data_config()
    data_loaded = False
    if data_config and params.get("databricks_share_name") and params.get("table_name"):
        print(f"\n[{i}] Processing datasets...")
        share_name = _set_share_name(
            user_email,
            params.get("databricks_share_name"),
            params.get("vocareum_share_name"),
        )
        process_csv_data(share_name, params.get("table_name"))
        create_sample_labels("tko_sample_labels")
        data_loaded = True
        i += 1

    # Resolve agent config early — needed to gate the agent_tools_dir crawl below.
    agent_cfg_section = cfg.get_agent_config()
    llm_endpoint = params.get("llm_endpoint_name")

    # ------------------------------------------------------------------
    # [i] SQL UC tool registration
    # Two sources (both run when configured):
    #   1. Config-specified tool names → built-in Includes_v2/tools/sql/ DDL files
    #   2. Directory crawl of agent_tools_dir subfolders (only when agents are configured)
    # ------------------------------------------------------------------
    sql_tools: set = set()
    config_tool_names = cfg.get("tools") or []
    if config_tool_names:
        builtin_sql_dir = Path(__file__).parent.parent / "tools" / "sql"
        for tool_name in config_tool_names:
            tool_path = builtin_sql_dir / f"{tool_name}.txt"
            if tool_path.exists():
                sql_tools.add((tool_name, builtin_sql_dir))
            else:
                print(f"  Warning: SQL tool DDL not found: {tool_path}")

    if agent_cfg_section and llm_endpoint and Path(agent_tools_dir).exists():
        sql_tools.update(tool_manager.discover_all_tools(agent_tools_dir))

    if sql_tools:
        print(f"\n[{i}] Registering {len(sql_tools)} SQL UC tool(s)...")
        tool_manager.create_tools(sql_tools)
        i += 1

    # ------------------------------------------------------------------
    # [i] Python UC tool registration
    # ------------------------------------------------------------------
    python_tool_names = cfg.get("python_tools") or []
    if python_tool_names:
        print(f"\n[{i}] Registering {len(python_tool_names)} Python UC tool(s)...")
        _register_python_uc_tools(python_tool_names, catalog_name, schema_name)
        i += 1


    # ------------------------------------------------------------------
    # [i] PDF
    # ------------------------------------------------------------------
    pdf_path = None
    if data_loaded and volume_path:
        print(f"\n[{i}] Creating Airbnb listings PDF...")
        pdf_output_path = str(volume_path / "airbnb_listings.pdf")
        pdf_path = create_listings_pdf(
            table_name=params.get("table_name"),
            output_path=pdf_output_path,
        )
        i += 1

    # ------------------------------------------------------------------
    # [i] Genie space
    # ------------------------------------------------------------------
    genie_space_id = None
    genie_config = cfg.get_genie_config()
    if genie_config and data_loaded and params.get("warehouse_name"):
        print(f"\n[{i}] Refreshing Genie space...")
        user_prefix = user_email.split("@")[0].replace(".", "_") + "_"
        if not user_prefix.startswith("labuser"):
            user_prefix = "labuser_" + user_prefix
        genie_space_name = user_prefix + params.get("genie_space_title", "")
        delete_genie_space_by_title(genie_space_name)
        genie_table = params.get("genie_source_table") or params.get("table_name")
        source_table = f"{catalog_name}.{schema_name}.{genie_table}"
        genie_space_id = create_genie_space(
            source_table=source_table,
            warehouse_name=params.get("warehouse_name"),
            space_title=genie_space_name,
            space_description=params.get("genie_space_description"),
        )
        i += 1

    # ------------------------------------------------------------------
    # [i] Agent setup (only when llm_endpoint_name is configured)
    # ------------------------------------------------------------------
    agent_configs_result = None
    deployed_endpoint = None

    if agent_cfg_section and llm_endpoint:
        print(f"\n[{i}] Setting up agents...")

        artifacts_dir = _includes_dir / "artifacts"
        configs_dir = artifacts_dir / "configs"

        artifacts_manager = ArtifactsManager(
            artifacts_dir=artifacts_dir,
            volume_path=volume_path,
        )
        config_renderer = ConfigRenderer(artifacts_dir=configs_dir)
        experiment_manager = ExperimentManager(username=user_email)

        lakebase_autoscaling_project = _set_lakebase_project_name(user_email)

        agent_manager = AgentManager(
            catalog_name=catalog_name,
            schema_name=schema_name,
            llm_endpoint_name=llm_endpoint,
            alias=params.get("alias", "champion"),
            username=user_email,
            artifacts_dir=configs_dir,
            eval_config_output_path=config_renderer.eval_config_output_path,
            lakebase_instance_name=params.get("lakebase_instance_name"),
            lakebase_autoscaling_project=lakebase_autoscaling_project,
            lakebase_autoscaling_branch=params.get("lakebase_autoscaling_branch"),
        )

        print(f"\n[{i}a] Creating artifacts structure...")
        artifacts_manager.copy_py_files_with_structure(
            py_source_dir=agents_dir,
            configs_source_dir=agent_configs_dir,
            eval_datasets_source_dir=eval_datasets_dir,
        )

        print(f"\n[{i}b] Discovering agent components...")
        agent_files = agent_manager.get_agent_files(artifacts_dir)
        print(f"  Found {len(agent_files)} agent file(s): {[f.name for f in agent_files]}")
        for agent_file in agent_files:
            agent_manager.map_agent_config(
                agent_file,
                tool_count_func=tool_manager.count_tool_placeholders,
                agent_configs_dir=agent_configs_dir,
            )

        # Also pick up config files that have no corresponding agent .py file.
        # This supports labs where the config is needed before the agent code exists.
        already_mapped = {
            acfg["config_template"].name
            for acfg in agent_manager.agent_configs.values()
        }
        for config_path in agent_manager.get_agent_config_files(agent_configs_dir):
            if config_path.name not in already_mapped:
                print(f"  Found config without agent .py: {config_path.name}")
                agent_manager.map_config_only(
                    config_path,
                    tool_count_func=tool_manager.count_tool_placeholders,
                )

        print(f"\n[{i}c] Mapping tools to agents...")
        for agent_name, acfg in agent_manager.agent_configs.items():
            tools = tool_manager.get_tools_for_agent(
                agent_name=agent_name,
                agent_tools_base_dir=agent_tools_dir,
            )
            acfg["tools"] = tools
            if tools:
                print(f"  - {agent_name}: {len(tools)} tool(s) → {tools}")
            else:
                print(f"  - {agent_name}: no tools found in '{agent_tools_dir}/{agent_name}'")

        eval_endpoints = {
            "correctness":              params.get("correctness_eval_endpoint"),
            "retrieval_sufficiency":    params.get("retrieval_sufficiency_endpoint"),
            "custom":                   params.get("custom_eval_endpoint"),
            "guidelines":               params.get("guidelines_endpoint"),
            "recall":                   params.get("recall_eval_endpoint"),
        }
        if any(eval_endpoints.values()):
            print(f"\n[{i}d] Rendering evaluation config...")
            config_renderer.render_eval_config(
                correctness_endpoint=eval_endpoints["correctness"],
                retrieval_sufficiency_endpoint=eval_endpoints["retrieval_sufficiency"],
                guidelines_endpoint=eval_endpoints["guidelines"],
                custom_endpoint=eval_endpoints["custom"],
                recall_endpoint=eval_endpoints["recall"],
                template_dir=agent_configs_dir,
            )

        print(f"\n[{i}e] Registering agents...")
        for agent_name, acfg in agent_manager.agent_configs.items():
            print(f"\n  Registering agent: {agent_name}")
            substitutions = {
                "LLM_ENDPOINT_NAME": llm_endpoint,
                "CATALOG_NAME":      catalog_name,
                "SCHEMA_NAME":       schema_name,
            }
            if params.get("lakebase_instance_name"):
                substitutions["LAKEBASE_INSTANCE_NAME"] = params["lakebase_instance_name"]
            substitutions["LAKEBASE_AUTOSCALING_PROJECT"] = lakebase_autoscaling_project or ""
            substitutions["LAKEBASE_AUTOSCALING_BRANCH"]  = params.get("lakebase_autoscaling_branch") or ""
            for j, tool in enumerate(acfg.get("tools", []), start=1):
                substitutions[f"TOOL{j}"] = tool
            yaml_text = config_renderer.render_text_template(
                template_path=acfg["config_template"],
                substitutions=substitutions,
            )
            config_renderer.update_yaml_config(yaml_text, acfg["config_output"])
            print(f"  Config written: {acfg['config_output'].name}")
            experiment_path = experiment_manager.create_experiment(acfg["experiment_name"])
            agent_manager.register_agent(agent_name, acfg, acfg.get("tools", []), experiment_path)

        if agent_manager.agent_configs and params.get("deploy"):
            first_agent = list(agent_manager.agent_configs.keys())[0]
            deployment = agent_manager.deploy_agent(first_agent)
            deployed_endpoint = getattr(deployment, "endpoint_name", None)
            if deployed_endpoint:
                print(f"\n  Agent '{first_agent}' deployed to endpoint: {deployed_endpoint}")

        additional_experiments = cfg.get("agents.additional_experiments") or []
        if additional_experiments:
            print(f"\n[{i}g] Creating additional experiments...")
            for exp_name in additional_experiments:
                experiment_manager.create_experiment(exp_name)

        agent_configs_result = agent_manager.agent_configs

        # Collect all experiment names created so cleanup can delete them
        _created_experiment_names = [
            acfg["experiment_name"] for acfg in agent_configs_result.values()
        ] + list(additional_experiments)
    else:
        _created_experiment_names = []

    i += 1

    # ------------------------------------------------------------------
    # [i] Resource cleanup (dev_mode + resource_cleanup only)
    # ------------------------------------------------------------------
    _created_folders = [Path(f"./{params.get('folder')}")] if params.get("folder") else []
    dev_classroom_cleanup(
        dev_mode=dev_mode,
        resource_cleanup=resource_cleanup,
        catalog_name=catalog_name,
        step_index=i,
        artifacts_dir=_includes_dir / "artifacts",
        created_folders=_created_folders,
        experiment_names=_created_experiment_names,
        genie_space_title=params.get("genie_space_title"),
        user_email=user_email,
    )


    print("\n" + "=" * 60)
    print("Demo Environment Setup Complete")
    print("=" * 60)

    return {
        "catalog_name":            catalog_name,
        "schema_name":             schema_name,
        "volume_path":             str(volume_path) if volume_path else None,
        "username":                user_email,
        "table_name":              params.get("table_name"),
        "pdf_path":                pdf_path,
        "genie_space_id":          genie_space_id,
        "agent_configs":                agent_configs_result,
        "deployed_endpoint":            deployed_endpoint,
        "lakebase_instance_name":       params.get("lakebase_instance_name"),
        "lakebase_autoscaling_project": lakebase_autoscaling_project if agent_cfg_section and llm_endpoint else None,
        "lakebase_autoscaling_branch":  params.get("lakebase_autoscaling_branch"),
    }
