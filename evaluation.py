import os
import argparse
import shutil
import subprocess


def verify(path_to_run, command):
    try:
        process = subprocess.run(
            command,
            shell=True,
            cwd=path_to_run,
            timeout=180,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return "timeout", 1
    return process, 0


def evaluate(benchmark_language, test_project, verification):
    file_dir = f"/root/vert/benchmark/{benchmark_language}_transcoder"
    for subdir, _, files in os.walk(f"{file_dir}/"):
        for file in files:
            package_name = subdir.split("/")[-1]
            if "out-rwasm" in subdir:
                continue
            if (
                "test" in file
                or "processed" in file
                or "mutated" in file
                or "main" in file
                or "_diff" in file
                or "toml" in file
                or ".rs.c" in file
                or "_towasm" in file
                or "Cargo" in file
                or "CACHEDIR" in file
            ):
                continue
            if "src" in file or "src" in package_name or "out-rwasm" in package_name:
                continue
            if 'all_project' not in test_project and test_project != package_name:
                continue
            print(package_name)

            wasm_bolero_path = f"{file_dir}/{package_name}/out-rwasm-bolero/src"
            wasm_kani_path = f"{file_dir}/{package_name}/out-rwasm-mutated/src"
            bolero_target_path = wasm_bolero_path + "/target"
            kani_target_path = wasm_kani_path.replace("/src", "/target")
            #########################################################################################
            ###################################### BOLERO ###########################################
            #########################################################################################
            if verification == "bolero":
                print("Running bolero")
                command = "cargo bolero reduce bolero_wasm_eq"
                verification_output, timeout = verify(wasm_bolero_path, command)
                if not timeout:
                    err_message = verification_output.stderr
                    stdout_message = verification_output.stdout
                    if "could not compile" in err_message:
                        print("Compilation problem")
                    elif (
                        "Test Failure" in err_message
                        or "Test Failure" in stdout_message
                    ):
                        print("Bolero failed")
                    else:
                        print("Bolero pass")
                else:
                    print("Bolero timeout")
                shutil.rmtree(bolero_target_path)
            #########################################################################################
            ###################################### BoundedKANI ######################################
            #########################################################################################
            if verification == "bounded_kani":
                print("Running bounded Kani")
                command = "cargo kani --no-unwinding-checks --default-unwind 10"
                verification_output, timeout = verify(wasm_kani_path, command)
                if not timeout:
                    err_message = verification_output.stderr
                    stdout_message = verification_output.stdout
                    if (
                        "VERIFICATION:- FAILED" in err_message
                        or "VERIFICATION:- FAILED" in stdout_message
                    ):
                        print("Kani failed")
                    else:
                        print("Kani succesful")
                else:
                    print("Kani timeout")
                shutil.rmtree(kani_target_path)
            #######################################################################################
            ###################################### Full KANI ######################################
            #########################################################################################
            if verification == "full_kani":
                print("Running Kani")
                command = "cargo kani"
                verification_output, timeout = verify(wasm_kani_path, command)
                if not timeout:
                    err_message = verification_output.stderr
                    stdout_message = verification_output.stdout
                    if (
                        "VERIFICATION:- FAILED" in err_message
                        or "VERIFICATION:- FAILED" in stdout_message
                    ):
                        print("Kani failed")
                    else:
                        print("Kani succesful")
                else:
                    print("Kani timeout")
                shutil.rmtree(kani_target_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluation for VERT")
    ap.add_argument(
        "language",
        choices=["c", "cpp", "go"],
        help="Choose source language to compile to Rust: c, cpp, or go",
    )
    ap.add_argument("project", help="Provide a specific benchmark project name")
    ap.add_argument(
        "verification",
        choices=["bolero", "bounded_kani", "full_kani"],
        help="Verification type: bolero, bounded_kani, or full_kani",
    )
    args = ap.parse_args()


    language = args.language
    project = args.project
    verification = args.verification
    
    print(project)

    evaluate(language, project, verification)
