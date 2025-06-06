import EPTHenOpt

if __name__ == "__main__":
    # Run an optimization by passing parameters as keyword arguments
    EPTHenOpt.run(
        model='GA',
        epochs=10,
        population_size=50,
        number_of_workers=4,
        noverbose=True # Example of disabling verbose output
    )

    # You can also run a TLBO optimization
    print("\nStarting a new TLBO run...\n")
    EPTHenOpt.run(
        model='TLBO',
        epochs=5,
        population_size=40
    )