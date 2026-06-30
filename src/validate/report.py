"""Generate Markdown and JSON validation reports."""

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger('validate.report')


class ValidationReport:
    """Write a human-readable Markdown report and a JSON summary from
    characterization results.

    The report includes a header with run identification, tables for each
    validation domain (porosity, microstructure, stress and prediction
    comparison), and a pass/fail checklist derived from the available results.
    """

    def __init__(self, results, run_dir, run_id=None, lineage=None):
        """Initialize the report generator.

        Args:
            results: Dictionary returned by
                ``LPBFCharacterisation.run_full_characterisation``.
            run_dir: Directory where ``report.md`` and ``summary.json`` will
                be written.
            run_id: Optional run identifier. If omitted, a timestamp-based
                identifier is generated.
            lineage: Optional dictionary with dataset hashes or lineage
                information (e.g. ``{'xct_hash': 'abc...'}``).
        """
        self.results = results or {}
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name
        self.lineage = lineage or {}

    def write(self):
        """Generate and write the Markdown report and JSON summary.

        Returns:
            Tuple of (report_path, summary_path).
        """
        summary = self._build_summary()
        report_path = self._write_markdown(summary)
        summary_path = self._write_json(summary)
        logger.info("Validation report written to %s and %s", report_path, summary_path)
        return report_path, summary_path

    def _build_summary(self):
        """Build a flat summary dictionary from the characterization results."""
        summary = {
            'run_id': self.run_id,
            'lineage': self.lineage,
            'checklist': self._build_checklist(),
        }

        if 'porosity' in self.results:
            summary['porosity'] = self._flatten(self.results['porosity'])

        if 'microstructure' in self.results:
            summary['microstructure'] = self._flatten(self.results['microstructure'])

        if 'stress' in self.results:
            summary['stress'] = self._flatten(self.results['stress'])

        if 'comparison' in self.results:
            summary['comparison'] = self._flatten(self.results['comparison'])

        return summary

    @staticmethod
    def _flatten(value):
        """Recursively flatten nested dictionaries to JSON-safe primitives."""
        if isinstance(value, dict):
            flat = {}
            for key, item in value.items():
                flat[key] = ValidationReport._flatten(item)
            return flat
        if isinstance(value, list):
            return [ValidationReport._flatten(item) for item in value]
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        # Convert numpy/pandas scalars to Python primitives
        try:
            return float(value)
        except Exception:
            return str(value)

    def _build_checklist(self):
        """Build a checklist of validation items and their pass/fail status."""
        checklist = {}

        checklist['xct_porosity'] = 'porosity' in self.results
        checklist['ebsd_microstructure'] = 'microstructure' in self.results
        checklist['residual_stress'] = 'stress' in self.results
        checklist['prediction_comparison'] = 'comparison' in self.results

        if 'comparison' in self.results:
            comparison = self.results['comparison']
            for quantity in ['stress', 'porosity', 'geometric_accuracy']:
                if quantity in comparison:
                    error = comparison[quantity].get('error_percent', float('inf'))
                    checklist[f'{quantity}_within_tolerance'] = error < 20.0

        return checklist

    def _write_json(self, summary):
        """Write the JSON summary file."""
        summary_path = self.run_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary_path

    def _write_markdown(self, summary):
        """Write the Markdown report file."""
        report_path = self.run_dir / 'report.md'

        lines = []
        lines.append('# LPBF Validation Report')
        lines.append('')
        lines.append(f'**Run ID:** `{self.run_id}`')
        lines.append('')

        # Lineage / dataset hashes
        lines.append('## Dataset Lineage')
        lines.append('')
        if self.lineage:
            for key, value in self.lineage.items():
                lines.append(f'- **{key}:** `{value}`')
        else:
            lines.append('- No lineage information provided.')
        lines.append('')

        # Porosity table
        if 'porosity' in summary:
            lines.append('## Porosity (XCT)')
            lines.append('')
            lines.extend(self._dict_to_markdown_table(summary['porosity']))
            lines.append('')

        # Microstructure table
        if 'microstructure' in summary:
            lines.append('## Microstructure (EBSD)')
            lines.append('')
            lines.extend(self._dict_to_markdown_table(summary['microstructure']))
            lines.append('')

        # Stress table
        if 'stress' in summary:
            lines.append('## Residual Stress')
            lines.append('')
            lines.extend(self._dict_to_markdown_table(summary['stress']))
            lines.append('')

        # Prediction comparison table
        if 'comparison' in summary:
            lines.append('## Prediction Comparison')
            lines.append('')
            lines.extend(self._comparison_to_markdown_table(summary['comparison']))
            lines.append('')

        # Checklist
        lines.append('## Validation Checklist')
        lines.append('')
        lines.append('| Validation Item | Status |')
        lines.append('|-----------------|--------|')
        for item, passed in summary['checklist'].items():
            status = 'PASS' if passed else 'FAIL'
            lines.append(f'| {item} | {status} |')
        lines.append('')

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        return report_path

    @staticmethod
    def _dict_to_markdown_table(data):
        """Convert a nested dictionary into Markdown table rows.

        Args:
            data: Nested dictionary of metrics.

        Returns:
            List of Markdown table lines.
        """
        rows = []
        rows.append('| Metric | Value |')
        rows.append('|--------|-------|')

        def add_rows(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    add_rows(f'{prefix}{key} ', item)
            elif isinstance(value, list):
                rows.append(f'| {prefix.strip()} | {len(value)} items |')
            else:
                rows.append(f'| {prefix.strip()} | {value} |')

        add_rows('', data)
        return rows

    @staticmethod
    def _comparison_to_markdown_table(comparison):
        """Convert prediction-comparison metrics into a Markdown table.

        Args:
            comparison: Dictionary of comparison metrics.

        Returns:
            List of Markdown table lines.
        """
        rows = []
        rows.append('| Quantity | Predicted | Experimental | MAE | RMSE | Error % |')
        rows.append('|----------|-----------|--------------|-----|------|---------|')

        for quantity in ['stress', 'porosity', 'geometric_accuracy']:
            if quantity not in comparison:
                continue
            metrics = comparison[quantity]
            rows.append(
                f"| {quantity} | "
                f"{metrics.get('predicted', 'N/A')} | "
                f"{metrics.get('experimental', 'N/A')} | "
                f"{metrics.get('mae', 'N/A')} | "
                f"{metrics.get('rmse', 'N/A')} | "
                f"{metrics.get('error_percent', 'N/A')} |"
            )

        if 'r2_score' in comparison:
            rows.append(f"| R² score | {comparison['r2_score']} | | | | |")

        return rows


def main():
    """CLI entry point to regenerate a report from a YAML results file."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate validation report')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to characterization results YAML file')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Directory to write report.md and summary.json')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Optional run identifier')

    args = parser.parse_args()

    with open(args.results, 'r') as f:
        results = yaml.safe_load(f)

    report = ValidationReport(results, args.run_dir, run_id=args.run_id)
    report.write()
    print(f"Report written to {args.run_dir}")


if __name__ == '__main__':
    main()
