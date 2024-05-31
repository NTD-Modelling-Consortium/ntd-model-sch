from argparse import ArgumentParser

from .helsim_RUN_KK import SCH_Simulation


def parse_arguments():
    desc = "NTDMC SCH model command line interface"
    parser = ArgumentParser(description=desc)

    parser.add_argument(
        "-p", "--parameter-set", type=str,
        help="The set of simulation parameters",
        choices=["high-adult-burden", "low-adult-burden"],
        default="low-adult-burden",
    )
    parser.add_argument(
        "-d", "--demog-name", type=str,
        help="Name of demography to simulate for",
        choices=[
            "WHOGeneric",
            "UgandaRural",
            "KenyaKDHS",
            "Flat",
        ],
        default="WHOGeneric",
    )
    return parser


def main(arguments=None):
    parser = parse_arguments()
    args = parser.parse_args(arguments)
    param_file_name = (
        "SCH-" +
        args.parameter_set.replace("-", "_") +
        ".txt"
    )

    df = SCH_Simulation(
        paramFileName=param_file_name,
        demogName=args.demog_name,
        numReps=10,
    )
    df.to_json('sch_results.json')

    df.plot(
        x='Time',
        y=['SAC Prevalence', 'Adult Prevalence']
    )

    df.plot(
        x='Time',
        y=[
            'SAC Heavy Intensity Prevalence',
            'Adult Heavy Intensity Prevalence'
        ]
    )
