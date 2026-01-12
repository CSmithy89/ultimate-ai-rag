from __future__ import annotations

from dataclasses import dataclass

from rich.prompt import Confirm, Prompt

from cli.prompts.shared import validate_api_key


@dataclass
class FastPathSelections:
    profile: str
    llm_provider: str
    api_key: str
    framework: str
    customize: bool


def run_fast_path(
    console,
    recommended_profile: str,
    llm_providers: list[str],
    frameworks: list[str],
) -> FastPathSelections:
    accept_recommended = Confirm.ask(
        f"Accept recommended profile? ({recommended_profile})",
        default=True,
        console=console,
    )
    if accept_recommended:
        profile = recommended_profile
    else:
        profile = Prompt.ask(
            "Select profile",
            choices=["minimal", "standard", "enterprise"],
            default=recommended_profile,
            console=console,
        )

    llm_provider = Prompt.ask(
        "LLM Provider",
        choices=llm_providers,
        default=llm_providers[0],
        console=console,
    )

    api_key = ""
    if validate_api_key(llm_provider, "") is False:
        api_key = Prompt.ask(
            f"Enter {llm_provider} API key",
            password=True,
            console=console,
        ).strip()
        while not validate_api_key(llm_provider, api_key):
            console.print("Invalid key format. Please try again.")
            api_key = Prompt.ask(
                f"Enter {llm_provider} API key",
                password=True,
                console=console,
            ).strip()

    framework = Prompt.ask(
        "Generate framework starter?",
        choices=frameworks,
        default="none",
        console=console,
    )

    proceed = Prompt.ask(
        "Proceed with install?",
        choices=["y", "n", "c"],
        default="y",
        console=console,
    )
    if proceed == "n":
        raise SystemExit(0)
    customize = proceed == "c"

    return FastPathSelections(
        profile=profile,
        llm_provider=llm_provider,
        api_key=api_key,
        framework=framework,
        customize=customize,
    )
