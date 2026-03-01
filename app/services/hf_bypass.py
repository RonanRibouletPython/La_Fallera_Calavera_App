from pathlib import Path

import transformers.dynamic_module_utils

_original_get_imports = transformers.dynamic_module_utils.get_imports


def _get_imports_without_flash_attn(filename: str | Path) -> list[str]:
    """Filter out flash_attn from Hugging Face's static import analysis."""
    imports = _original_get_imports(filename)
    return [imp for imp in imports if imp != "flash_attn"]


# Apply the monkey-patch
transformers.dynamic_module_utils.get_imports = _get_imports_without_flash_attn
