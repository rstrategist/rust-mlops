#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;

#[test]
fn scripts_are_executable() {
    let scripts = [
        "scripts/run_translate.sh",
        "scripts/run_heavy_translate.sh",
        "scripts/build-with-libtorch.sh",
    ];

    for s in scripts {
        let meta = fs::metadata(s).unwrap_or_else(|_| panic!("Expected {} to exist", s));
        let mode = meta.permissions().mode();
        assert!(
            mode & 0o111 != 0,
            "Script {} is not executable (mode: {:o})",
            s,
            mode
        );
    }
}
