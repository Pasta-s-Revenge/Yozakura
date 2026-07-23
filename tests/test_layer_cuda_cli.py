from yozakura.cli import _parse_max_memory, _parser


def test_parse_max_memory_accepts_cuda_and_cpu_budgets() -> None:
    assert _parse_max_memory(["0=8GiB", "cpu=6GiB"]) == {0: "8GiB", "cpu": "6GiB"}


def test_layer_help_documents_cuda_budget(capsys) -> None:
    parser = _parser()

    try:
        parser.parse_args(["run", "model.sun", "--prompt", "test", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert "CUDA-resident layer budget" in output
    assert "accepts both GPU and CPU budgets" in output
