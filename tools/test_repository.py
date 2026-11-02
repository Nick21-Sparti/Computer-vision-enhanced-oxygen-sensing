"""Lightweight regression checks for repository packaging; no research imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repo_config as cfg


def assigned_value(filename, name, namespace=None):
    source = (ROOT / "scripts" / filename).read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)
    scope = {"data_path": cfg.data_path, "output_path": cfg.output_path,
             "cache_path": cfg.cache_path, "external_path": cfg.external_path}
    scope.update(namespace or {})
    return eval(compile(ast.Expression(node.value), filename, "eval"), scope)


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_cwd_independent_defaults(self):
        before = cfg.data_path("aqueous")
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(before, cfg.data_path("aqueous"))
                self.assertEqual(Path(before), ROOT / "data" / "aqueous")
            finally:
                os.chdir(previous)

    def test_external_absolute_roots_and_run_names(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ.update(OXYGEN_AQUEOUS_DATA=directory, OXYGEN_OUTPUT_ROOT=directory,
                              OXYGEN_CACHE_ROOT=directory, OXYGEN_RUN_NAME="trial_2")
            self.assertEqual(Path(cfg.data_path("aqueous")), Path(directory))
            self.assertEqual(Path(cfg.output_path("aqueous_tvit")), Path(directory)/"trial_2"/"aqueous_tvit")
            self.assertEqual(Path(cfg.cache_path("x", "features")), Path(directory)/"trial_2"/"x"/"features")
            os.environ["OXYGEN_RUN_NAME"] = "trial_3"
            self.assertIn("trial_3", cfg.output_path("x"))
            self.assertIn("trial_3", cfg.cache_path("x"))

    def test_invalid_run_name_rejected(self):
        for value in ["../old", "a/b", "..", "a b"]:
            os.environ["OXYGEN_RUN_NAME"] = value
            with self.assertRaises(ValueError):
                cfg.output_path("x")

    def test_optional_inputs_and_roi_follow_data_root(self):
        self.assertIsNone(cfg.external_path("gas_frame_cache"))
        with tempfile.TemporaryDirectory() as directory:
            os.environ["OXYGEN_GAS_DATA"] = directory
            self.assertEqual(Path(cfg.external_path("gas_roi")), Path(directory)/"global_roi_settings.json")
            os.environ["OXYGEN_GAS_FRAME_CACHE"] = str(Path(directory)/"frames.parquet")
            self.assertEqual(Path(cfg.external_path("gas_frame_cache")), Path(directory)/"frames.parquet")

    def test_analysis_outputs_do_not_collide(self):
        outputs=[]
        for path in (ROOT/"scripts").glob("*.py"):
            for name in ["OUTPUT_DIR", "BASE_OUTPUT"]:
                try:
                    value=assigned_value(path.name,name)
                except StopIteration:
                    continue
                outputs.append(value)
        self.assertEqual(len(outputs),len(set(outputs)))

    def test_shared_heterogeneity_routes(self):
        names=["heterogeneity_quality_groups.py", "heterogeneity_group_size.py", "heterogeneity_linear_baselines.py"]
        paths=[assigned_value(name,"FEATURE_CACHE_DIR") for name in names]
        self.assertEqual(len(set(paths)),1)
        info=assigned_value("heterogeneity_information_groups.py","FEATURES_PATH")
        self.assertEqual(Path(info),Path(paths[0])/"idc_master_v10.parquet")
        quality=assigned_value(names[0],"STUDY_DIR")
        linear=assigned_value(names[2],"STUDY_DIR")
        self.assertEqual(quality,linear)
        features=assigned_value("heterogeneity_information_groups.py","STUDY_FEATURES")
        self.assertEqual(Path(features),Path(quality)/"Group_Definitions"/"pixel_features.npz")

    def test_gas_equilibrium_routes(self):
        # Execute only the path resolver in an isolated temporary output root.
        with tempfile.TemporaryDirectory() as directory:
            os.environ["OXYGEN_OUTPUT_ROOT"] = directory
            for key in ["gas_equilibrium_stern_volmer", "gas_equilibrium_polynomial"]:
                path=ROOT/"scripts"/(key+".py")
                fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="resolve_paths")
                namespace={"P":{},"os":os,"output_path":cfg.output_path,"external_path":cfg.external_path}
                exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace)
                namespace["resolve_paths"]()
                self.assertEqual(namespace["P"]["v11_cache"],cfg.output_path("gas_tvit","4_Caches"))
                self.assertEqual(namespace["P"]["v14_cache"],cfg.external_path("gas_posthoc_cache"))
                self.assertEqual(namespace["P"]["out"],cfg.output_path(key))

    def test_all_sources_and_source_manifest(self):
        manifest=json.loads((ROOT/"docs"/"SOURCE_MANIFEST.json").read_text())
        self.assertEqual(len(manifest),16)
        for item in manifest:
            path=ROOT/item["repository_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item["repository_sha256"])
            compile(path.read_text(),str(path),"exec")
            self.assertTrue(any(isinstance(n,ast.If) and "__name__" in ast.unparse(n.test) for n in ast.parse(path.read_text()).body))


if __name__ == "__main__":
    unittest.main()
