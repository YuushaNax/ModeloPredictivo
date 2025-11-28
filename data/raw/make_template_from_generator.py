import argparse
import runpy
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Create an empty template XLSX using an existing generator script')
    parser.add_argument('--generator', default='data/raw/generator_empty.py', help='Path to the generator script (must define generate)')
    parser.add_argument('--output', default='template_empty.xlsx', help='Output xlsx template path')
    parser.add_argument('--strip-results', action='store_true', help='Remove result columns from the template (input-only)')
    args = parser.parse_args()

    gen_path = Path(args.generator)
    if not gen_path.exists():
        print(f'Generator script not found: {gen_path}')
        return

    # Execute the generator script to get its globals (including generate function)
    g = runpy.run_path(str(gen_path))
    if 'generate' not in g:
        print('The generator script does not define a `generate(n, out_path)` function')
        return

    generate = g['generate']
    # Call generate with n=0 to produce an empty dataframe (columns only)
    tmp_out = args.output
    try:
        generate(n=0, out_path=tmp_out)
    except Exception as e:
        print(f'Failed to run generator: {e}')
        return

    # If requested, remove the result columns and rewrite
    if args.strip_results:
        try:
            df = pd.read_excel(tmp_out)
            result_cols = ['Probabilidad de accidente', 'Accidente', 'Condicion del Vehiculo']
            cols = [c for c in df.columns if c not in result_cols]
            df2 = pd.DataFrame(columns=cols)
            df2.to_excel(tmp_out, index=False)
            print(f'Input-only template written to {tmp_out}')
        except Exception as e:
            print(f'Failed to strip result columns: {e}')
    else:
        print(f'Template written to {tmp_out}')


if __name__ == '__main__':
    main()
