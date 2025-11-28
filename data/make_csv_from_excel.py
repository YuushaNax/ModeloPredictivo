import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .xlsx file")
    parser.add_argument("--output", required=True, help="Output .csv file")
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    df.to_csv(args.output, index=False)
    print(f"Wrote CSV: {args.output}")


if __name__ == '__main__':
    main()
