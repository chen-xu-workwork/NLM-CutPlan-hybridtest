import pathlib
import tempfile
import unittest

from hybrid_planner.prompting.builder import (
    HybridPromptBuilder,
    PromptBuilderConfig,
)


class PromptBuilderTests(unittest.TestCase):
    def test_runtime_translation_returns_text_without_output_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            problem_dir = root / "problems"
            problem_dir.mkdir()
            domain_path = root / "domain.pddl"
            domain_path.write_text("(define (domain demo))", encoding="utf-8")
            problem_path = problem_dir / "problem-1.pddl"
            problem_path.write_text(
                "(define (problem p)\n"
                "  (:domain demo)\n"
                "  (:init (old))\n"
                "  (:goal (new))\n"
                ")\n",
                encoding="utf-8",
            )
            domain_code_path = root / "domain.txt"
            domain_code_path.write_text("domain manual", encoding="utf-8")

            builder = HybridPromptBuilder(
                PromptBuilderConfig(
                    domain_pddl=domain_path,
                    problem_dir=problem_dir,
                    domain_code=domain_code_path,
                )
            )
            translated_paths = []

            def translate_problem(problem_file_path):
                translated_paths.append(pathlib.Path(problem_file_path))
                return "unused Python source", "translated problem"

            builder._translate_problem = translate_problem
            builder._system_template = "system {domain_code}"
            builder._user_template = "user {problem_description}"

            built = builder.build("problem-1", "(:init (current))")

            self.assertEqual(len(translated_paths), 1)
            self.assertIn("(:init (current))", built.runtime_problem)
            self.assertNotIn("(:init (old))", built.runtime_problem)
            self.assertEqual(built.problem_description, "translated problem")
            self.assertIn("domain manual", built.system)
            self.assertEqual(built.user, "user translated problem")


if __name__ == "__main__":
    unittest.main()
