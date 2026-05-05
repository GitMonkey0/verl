from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in ["math_dapo", "math", "math_dapo_reasoning"] or str(data_source).startswith("aime"):
        from verl.utils.reward_score import math_dapo

        return math_dapo.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            strict_box_verify=True,
        )

    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
