nextflow.enable.dsl = 2

baseDir = projectDir

params.conda = "/home/jgauthie/.conda/envs/huggingface"
params.outputDir = "output"

params.targetLength = 500
params.targetSize = 2000

params.modelId = "gpt2"


supportedSuites = channel.from(
    "number_prep", "number_src", "number_orc",
    "reflexive_src_fem", "reflexive_src_masc",
    "reflexive_orc_fem", "reflexive_orc_masc",
    "reflexive_prep_fem", "reflexive_prep_masc",
    "subordination_src-src", "subordination_orc-orc", "subordination_pp-pp",
    "mvrr",
    "fgd_pp", "fgd_subject", "fgd_object"
)

process evaluatePrefixes {
    conda params.conda

    publishDir "${params.outputDir}"
    tag "${suite}/${prefix_suite}"

    // aws.io causes an abort and core dump on termination, yikes.
    // just don't let it stop the flow ..
    errorStrategy 'ignore'

    input:
    tuple val(suite), val(prefix_suite)

    output:
    tuple file("${suite}.${prefix_suite}.predictions.csv"),
          file("${suite}.${prefix_suite}.regions.csv")

    script:
    """
    python ${baseDir}/analyze_repetitions.py \
        --suite ${suite} \
        --prefix_suite ${prefix_suite} \
        --output-file ${suite}.${prefix_suite} \
        --target-length ${params.targetLength} \
        --target-size ${params.targetSize} \
        --model-id ${params.modelId}

    # Remove dumped core :(
    rm -f core*
    """
}


workflow {
    supportedSuites.combine(supportedSuites) | evaluatePrefixes
}