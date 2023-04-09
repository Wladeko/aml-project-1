import pkg_resources

# Load requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Print current versions of packages in the system
for package in requirements:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package} is not installed")
