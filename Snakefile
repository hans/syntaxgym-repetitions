configfile: "config.yaml"


wildcard_constraints:
    prefix_type = "grammatical|ungrammatical"


rule all:
    input:
        expand("output/{model}/{prefix_type}/{suite}/{prefix_suite}",
               model=config["models"], prefix_type=["grammatical", "ungrammatical"],
               suite=config["suites"], prefix_suite=config["suites"])

rule evaluate_prefixes:
    output:
        directory("output/{model}/{prefix_type}/{suite}/{prefix_suite}")

    shell:
        """
        mkdir -p {output}

        python analyze_repetitions.py \
            --suite {wildcards.suite} \
            --prefix_suite {wildcards.prefix_suite} \
            --prefix_type {wildcards.prefix_type} \
            -o {output} \
            --target-length {config[prefixing][target_length]} \
            --target-size {config[prefixing][target_size]} \
            --model-id {wildcards.model} \
            || echo "core dump probably"

        # Remove dumped core :(
        rm -f core*
        """