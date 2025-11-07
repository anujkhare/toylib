import internal1
import external1
import external2

# import local modules
from internal2.foo import data
from internal2.bar import model
from internal2.bar import experiment


def foo():
    print(internal1.some_function())
    print(external1.some_external_function())
    print(data.load_data())
    print(model.build_model())
    print(experiment.run_experiment())


def main():
    print("This is the entrypoint of the sample project.")
