configfile: "config.yaml"


rule all:
    input:
        expand("output/{model}/{suite}/{prefix_suite}",
               model=config["models"], suite=config["suites"],
               prefix_suite=config["suites"])

rule evaluate_prefixes:
    output:
        directory("output/{model}/{suite}/{prefix_suite}")

    shell:
        """
        python analyze_repetitions.py \
            --suite {wildcards.suite} \
            --prefix_suite {wildcards.prefix_suite} \
            --output-file {wildcards.suite}.{wildcards.prefix_suite} \
            --target-length {config[prefixing][target_length]} \
            --target-size {config[prefixing][target_size]} \
            --model-id {wildcards.model} \
            || echo "core dump probably"

        # Remove dumped core :(
        rm -f core*
        """