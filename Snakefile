configfile: "config.yaml"


# snakemake -p --cluster "sbatch -t 0-2 -p normal --gres=gpu:1 --constraint=any-A100 --mem 32G -n 1" -j 16

wildcard_constraints:
    prefix_type = "grammatical|ungrammatical"


rule all:
    input:
        expand("output/{model}/{prefix_type}/{suite}/{prefix_suite}",
               model=config["models"], prefix_type=["grammatical", "ungrammatical"],
               suite=config["suites"], prefix_suite=config["suites"])


rule evaluate_prefixes:
    resources:
        mem_mb = 10000
    output:
        directory("output/{model}/{prefix_type}/{suite}/{prefix_suite}")

    shell:
        """
        mkdir -p {output}

        bash -c '
            . $HOME/.bashrc # if not loaded automatically
            conda activate huggingface
        python analyze_repetitions.py \
            --suite {wildcards.suite} \
            --prefix_suite {wildcards.prefix_suite} \
            --prefix_type {wildcards.prefix_type} \
            -o {output} \
            --target-length {config[prefixing][target_length]} \
            --target-size {config[prefixing][target_size]} \
            --model-id {wildcards.model} \
            || echo "core dump probably"
        '

        # Remove dumped core :(
        rm -f core*
        """


rule evaluate_fixed_prefixes:
    resources:
        mem_mb = 10000

    input:
        prefix_file = "prefixes/{prefix_source}.json"

    output:
        directory("output/{model}/fixed/{suite}/{prefix_source}")

    shell:
        """
        mkdir -p {output}

        bash -c '
            . $HOME/.bashrc # if not loaded automatically
            conda activate huggingface
        python evaluate_with_prefixes.py \
            --suite {wildcards.suite} \
            --prefix_file {input.prefix_file} \
            --prefix_label {wildcards.prefix_source} \
            -o {output} \
            --target-lengths 20,100,200,300,400,500,600 \
            --target-size {config[prefixing][target_size]} \
            --model-id {wildcards.model} \
            || echo "core dump probably"
        '

        # Remove dumped core :(
        rm -f core*
        """