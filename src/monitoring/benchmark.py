"""
benchmark.py

Benchmark prediction latency.
Measures prediction performance over multiple runs.
"""

import sys
import time

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import (
    read_yaml,
    save_json
)


class ModelBenchmark:

    def __init__(self):

        try:

            self.config = read_yaml(
                "configs/config.yaml"
            )

            self.report_path = self.config["paths"][
                "benchmark_report"
            ]

            # Number of benchmark runs
            self.runs = self.config.get(
                "benchmark",
                {}
            ).get(
                "runs",
                10
            )

        except Exception as e:

            raise CustomException(e, sys)

    def measure(self, prediction_pipeline, sample_data):

        """
        Measure prediction latency over multiple runs.
        """

        try:

            logger.info(
                "Starting model benchmark..."
            )

            latencies = []

            # Run prediction multiple times
            for run_number in range(
                1,
                self.runs + 1
            ):

                start_time = time.perf_counter()

                prediction_pipeline.predict(
                    sample_data
                )

                end_time = time.perf_counter()

                latency = (
                    end_time - start_time
                )

                latencies.append(
                    latency
                )

                logger.info(
                    f"Benchmark run "
                    f"{run_number}/{self.runs}: "
                    f"{latency * 1000:.3f} ms"
                )

            # Calculate latency statistics
            total_latency = sum(
                latencies
            )

            average_latency = (
                total_latency
                / len(latencies)
            )

            minimum_latency = min(
                latencies
            )

            maximum_latency = max(
                latencies
            )

            # Create benchmark report
            report = {

                # Backward-compatible field
                # Required by the existing test
                "prediction_time_seconds":
                    round(
                        average_latency,
                        6
                    ),

                # Backward-compatible field
                # Required by the existing test
                "prediction_time_milliseconds":
                    round(
                        average_latency * 1000,
                        3
                    ),

                # Number of benchmark runs
                "benchmark_runs":
                    self.runs,

                # Average latency
                "average_prediction_time_seconds":
                    round(
                        average_latency,
                        6
                    ),

                "average_prediction_time_milliseconds":
                    round(
                        average_latency * 1000,
                        3
                    ),

                # Minimum latency
                "minimum_prediction_time_milliseconds":
                    round(
                        minimum_latency * 1000,
                        3
                    ),

                # Maximum latency
                "maximum_prediction_time_milliseconds":
                    round(
                        maximum_latency * 1000,
                        3
                    ),

                "status":
                    "Success"
            }

            # Save benchmark report
            save_json(
                self.report_path,
                report
            )

            logger.info(
                "Benchmark completed successfully."
            )

            logger.info(
                f"Average latency: "
                f"{report['average_prediction_time_milliseconds']} ms"
            )

            logger.info(
                f"Minimum latency: "
                f"{report['minimum_prediction_time_milliseconds']} ms"
            )

            logger.info(
                f"Maximum latency: "
                f"{report['maximum_prediction_time_milliseconds']} ms"
            )

            return report

        except Exception as e:

            logger.error(
                "Benchmark failed."
            )

            raise CustomException(
                e,
                sys
            )