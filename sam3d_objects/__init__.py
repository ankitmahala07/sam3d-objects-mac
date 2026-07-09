# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# Stub kaolin so imports succeed without the package installed.
from sam3d_objects._kaolin_stub import _install as _kaolin_install
_kaolin_install()

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    try:
        import sam3d_objects.init
    except ModuleNotFoundError as exc:
        if exc.name != "sam3d_objects.init":
            raise
