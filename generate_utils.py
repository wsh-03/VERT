import subprocess
import re
import difflib
import os, sys
from subprocess import STDOUT, check_output
from pythonds.basic import Stack


class VerificationUtils:
    def verify(self, path_to_run, command):
        try:
            process = subprocess.run(
                command,
                shell=True,
                cwd=path_to_run,
                timeout=200,
                text=True,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            return "timeout", 1
        return process, 0

    def mutate_test(
        self,
        file_dir,
        package_name,
        file_path,
        fn_name,
        args_types,
        file_ext,
        fn_out_type,
    ):
        if "cpp" in file_ext:
            cpp = "++"
        else:
            cpp = ""
        filename = file_path.split("/")[-1]
        if isinstance(args_types, str):
            args_types = args_types.split(", ")

        if " " in fn_name:
            fn_name = fn_name.split(" ")[-1]
        mutated_list = []

        project_path = f"{file_dir}/{package_name}"

        c_to_wasi = subprocess.run(
            f"/wasi-sdk-12.0/bin/clang{cpp} -fno-exceptions --sysroot=/wasi-sdk-12.0/share/wasi-sysroot -o main.wasm {filename}",
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_path,
        )

        wasi_to_rust = subprocess.run(
            f"cargo run --manifest-path /rWasm/Cargo.toml --release -- main.wasm out-rwasm-original --prevent-reformat --wasi-executable",
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_path,
        )

        c_to_wasi = subprocess.run(
            f"/wasi-sdk-12.0/bin/clang{cpp} -fno-exceptions --sysroot /wasi-sdk-12.0/share/wasi-sysroot -o main.wasm {filename.replace(file_ext, f'_mutated{file_ext}')}",
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_path,
        )
        wasi_to_rust = subprocess.run(
            f"cargo run --manifest-path /rWasm/Cargo.toml --release -- main.wasm out-rwasm-mutated --prevent-reformat --wasi-executable",
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_path,
        )
        wasi_to_rust = subprocess.run(
            f"cargo run --manifest-path /rWasm/Cargo.toml --release -- main.wasm out-rwasm-bolero --prevent-reformat --wasi-executable",
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_path,
        )

        original_rust_path = project_path + "/out-rwasm-original/src/main.rs"
        mutated_rust_path = project_path + "/out-rwasm-mutated/src/main.rs"
        bolero_rust_path = project_path + "/out-rwasm-bolero/src/main.rs"
        bolero_cargo = project_path + "/out-rwasm-bolero/Cargo.toml"
        with open(bolero_cargo, "a") as file:
            file.write('\n[dev-dependencies]\nbolero = "0.10.0"')

        with open(original_rust_path, "r") as file:
            original_rust = file.readlines()

        with open(mutated_rust_path, "r") as file:
            mutated_rust = file.readlines()

        with open(bolero_rust_path, "r") as file:
            bolero_rust = file.readlines()

        diff_list = []
        mutated_list = tuple(mutated_list)
        rwasm_arg_types = []
        target_mutation_lines = []
        diff_num = 0

        diff_flag = 1
        for line in difflib.unified_diff(
            original_rust,
            mutated_rust,
            fromfile="original",
            tofile="mutated",
            lineterm="",
        ):
            if "m.memory[" in line:
                break
            diff_list.append(line)
            if line.startswith("@@"):
                if diff_flag:
                    diff_num = int(line.split(" ")[1].split(",")[0].replace("-", ""))
                    diff_flag = 0
            if line.startswith("+v"):
                target_mutation_lines.append(line.replace("+", ""))
                type_extraction = line.split("(")[1]
                type_extraction = type_extraction.replace(");\n", "")
                for mutate in mutated_list:
                    type_extraction = type_extraction.replace(mutate, "")
                rwasm_arg_types.append(type_extraction)

        rwasm_arg_declaration = ""
        for i, arg_type in enumerate(args_types):
            arg_type = (
                arg_type.replace("unsigned int", "u32")
                .replace("int", "i32")
                .replace("double", "f32")
                .replace("float", "f32")
                .replace("string", "char")
            )
            param_insertion = "12"
            if "char" in arg_type:
                param_insertion = "'a'"
            elif "f32" in arg_type:
                param_insertion = "12.0"
            if "[]" in arg_type:
                rwasm_arg_declaration += f"static mut PARAM{i+1}: [{arg_type.replace('[]', '')}; 2] = [{param_insertion},{param_insertion}];\n"
            else:
                rwasm_arg_declaration += (
                    f"static mut PARAM{i+1}: {arg_type} = {param_insertion};\n"
                )

        result_insertion = "12"
        if "char" in fn_out_type or "string" in fn_out_type:
            result_insertion = "'c'"
            fn_out_type = "char"            
        elif "double" in fn_out_type or "float" in fn_out_type:
            fn_out_type = "f32"
            result_insertion = "12.0"
        elif "long" in fn_out_type:
            fn_out_type = "i32"
        elif "unsigned" in fn_out_type:
            fn_out_type = "u32"
        rwasm_arg_declaration += (
            f"static mut RESULT: {fn_out_type} = {result_insertion};\n"
        )

        diff = "\n".join(diff_list)
        diff_path = file_path.replace(file_ext, "_diff.txt")
        with open(diff_path, "w") as file:
            file.write(diff)

        func_lines = []
        for line_idx, line in enumerate(mutated_rust):
            if line.strip().startswith("fn func_"):
                func_lines.append(line_idx - 1)

        str_mutation = "".join(mutated_rust)

        if diff_num:
            window = []
            for i, line in enumerate(func_lines):
                if diff_num > line and diff_num < func_lines[i + 1]:
                    window = (line, func_lines[i + 1])
                    break
                elif i == 0 and diff_num < line:
                    func_num = int(len(func_lines) / 2)
                    window = (func_lines[func_num - 1] + 1, func_lines[func_num + 2])
            if window:
                target_function = original_rust[window[0] : window[1]]
                return_line = ""
                for line in target_function:
                    if "TaggedVal::from(self.func" in line:
                        return_line = line
                if "[]" in args_types[0]:
                    unsafe_param = "{PARAM1}[0]"
                    unsafe_param_kani = "PARAM1[0]"
                elif "char" in args_types[0] or "string" in args_types[0]:
                    unsafe_param = "{PARAM1 as i32}"
                    unsafe_param_kani = "PARAM1 as i32"
                else:
                    unsafe_param = "{PARAM1}"
                    unsafe_param_kani = "PARAM1"


                if 'char' in fn_out_type:
                    fn_out_type = 'i32'
                bolero_entry = str_mutation.replace(
                    return_line,
                    f"v0 = TaggedVal::from(unsafe {unsafe_param});\n"
                    + return_line
                    + f"\nlet retval = v0.try_as_{fn_out_type}()?;\nunsafe {{\nRESULT = retval;\n}}\n",
                )

                kani_entry = str_mutation.replace(
                    return_line,
                    f"v0 = TaggedVal::from(unsafe {{\n\t{unsafe_param_kani} = kani::any();\n\tkani::assume((0..2).contains(&{unsafe_param_kani}));\n\t{unsafe_param_kani}\n}});\n"
                    + return_line
                    + f"\nlet retval = v0.try_as_{fn_out_type}()?;\nunsafe {{\nRESULT = retval;\n}}\n",
                )
        kani_rust = rwasm_arg_declaration + kani_entry
        bolero_rust = rwasm_arg_declaration + bolero_entry
        print(kani_rust)
        print(bolero_rust)
        with open(mutated_rust_path, "w") as file:
            file.write(kani_rust)
        with open(bolero_rust_path, "w") as file:
            file.write(bolero_rust)
        subprocess.run(f"chmod -R a+rw {project_path}", shell=True)
        return rwasm_arg_types


class GenerateUtils:
    def build_rust_folder(self, rust_folder, package_name):
        subprocess.run(
            f"rm -rf {rust_folder}/{package_name}",
            shell=True,
            capture_output=True,
        )
        subprocess.run(
            f"cargo new --lib {package_name}",
            shell=True,
            cwd=rust_folder,
            capture_output=True,
        )
        subprocess.run(
            f"cargo bolero new {package_name}_test --generator; cargo add --dev bolero",
            shell=True,
            cwd=f"{rust_folder}/{package_name}",
            capture_output=True,
        )
        subprocess.run(
            f"chmod -R a+rw {package_name}",
            shell=True,
            cwd=rust_folder,
            capture_output=True,
        )

    def bracket_adder(self, code_string):
        code_string_split = re.split(r" |\n", code_string)
        curly_stack = Stack()
        parenthesis_stack = Stack()
        for i, char in enumerate(code_string_split):
            if not char or char == "\n":
                continue
            # print(str(i)  + ': ' + char.strip())

            if "{" in char.strip() and "}" in char.strip():
                pass
            elif "{" in char.strip():
                curly_stack.push(char)
            elif "}" in char.strip():
                try:
                    curly_stack.pop()
                except:
                    pass

            if "(" in char.strip():
                num = char.strip().count("(")
                for i in range(num):
                    parenthesis_stack.push(char)
            if ")" in char.strip():
                num = char.strip().count(")")
                for i in range(num):
                    try:
                        parenthesis_stack.pop()
                    except:
                        pass
        if not parenthesis_stack.isEmpty():
            for i in range(parenthesis_stack.size()):
                code_string += ")"
        if not curly_stack.isEmpty():
            for i in range(curly_stack.size()):
                code_string += "\n}"

        return code_string

    def remove_half_functions(self, input_text):
        input_text = input_text.replace("    ", "\t")
        split_input = input_text.split("\n")
        split_input_copy = split_input.copy()
        close_idx = 0
        for i, line in enumerate(split_input):
            if line.strip().startswith("func") or line.strip().startswith("fn"):
                has_close = False
                tab_count = line.count("\t")
                delete_start = i
                for j in range(i + 1, len(split_input)):
                    if split_input[j].count("\t") == tab_count:
                        has_close = True
                        close_idx = j
                        break
                    else:
                        delete_end = j
                if not has_close:
                    del split_input_copy[delete_start : delete_end + 1]
        split_input_copy = split_input_copy[: close_idx + 1]
        return "\n".join(split_input_copy)

    def keep_first_func(self, input_text):
        input_text = input_text.replace("    ", "\t")
        split_input = input_text.split("\n")
        split_input_copy = split_input.copy()
        close_idx = 0
        for i, line in enumerate(split_input):
            if line.strip().startswith("func") or line.strip().startswith("fn"):
                tab_count = line.count("\t")
                for j in range(i + 1, len(split_input)):
                    if (
                        split_input[j].count("\t") == tab_count and split_input[j]
                    ) and "}" in split_input[j]:
                        close_idx = j
                        break
            if close_idx:
                break
        if close_idx:
            split_input_copy = split_input_copy[: close_idx + 1]
        return "\n".join(split_input_copy)

    def check_comment(self, temp_code_line_nospace, is_block_comment):
        single_line_comment_flag = False
        comment_starts = ["/*", "'''", '"""', "```"]
        comment_ends = ["*/", "'''", '"""', "```"]
        single_line_comments = ["#", "*", "//", "/**"]
        if is_block_comment:
            if any(end_symbol in temp_code_line_nospace for end_symbol in comment_ends):
                single_line_comment_flag = True
                is_block_comment = False
                return single_line_comment_flag, is_block_comment
            else:
                single_line_comment_flag = True
                is_block_comment = True
                return single_line_comment_flag, is_block_comment
        elif (
            any(
                temp_code_line_nospace.startswith(single_line_comment)
                for single_line_comment in single_line_comments
            )
            or temp_code_line_nospace == "\n"
        ):
            single_line_comment_flag = True
            is_block_comment = False
            return single_line_comment_flag, is_block_comment
        elif any(
            temp_code_line_nospace.startswith(comment_start)
            for comment_start in comment_starts
        ):
            if any(end_symbol in temp_code_line_nospace for end_symbol in comment_ends):
                return True, False
            single_line_comment_flag = True
            is_block_comment = True
            return single_line_comment_flag, is_block_comment
        else:
            single_line_comment_flag = False
            is_block_comment = False
            return single_line_comment_flag, is_block_comment

    def remove_comments(self, code_string):
        target_lines = []
        is_block_comment = False
        for i, line in enumerate(code_string.split("\n")):
            is_comment, is_block_comment = self.check_comment(
                line.strip(), is_block_comment
            )
            if not is_comment and not is_block_comment and i > 0:
                target_lines.append(line)
        return "\n".join(target_lines)

    def error_msg_repair(self, rust_code, package_name, rust_dir, file_name):
        file_path = f"{rust_dir}/{package_name}/src/{file_name}.rs"
        # write to test rust file
        with open(file_path.replace(".rs", "_test.rs"), "w") as file:
            file.write(rust_code)
        no_errors = False
        iter = 0
        return_code = rust_code
        while not no_errors:
            # read test rust file
            with open(file_path.replace(".rs", "_test.rs"), "r") as file:
                rust_code = file.readlines()
            rust_output = subprocess.run(
                f"rustc {file_name}_test.rs",
                shell=True,
                capture_output=True,
                text=True,
                cwd=f"{rust_dir}/{package_name}/src",
            )

            # try 10 times
            if iter > 10:
                break
            iter += 1

            error_list = rust_output.stderr.splitlines()
            error_string = "\n".join(error_list)


            with open(file_path.replace(".rs", "_error.txt"), "w") as file:
                file.write(error_string)

            return_code = "".join(rust_code)
            if "error" not in error_string or rust_output.stdout:
                print("no more errors")
                with open(file_path.replace(".rs", "_compiles.rs"), "w") as file:
                    file.write(return_code)
                    no_errors = True
                break
            elif "consider" not in error_string:
                break

            help_flag = False
            suggestion_idx = ""
            suggestion_code = ""
            error_idx = ""
            error_code = ""
            for eline in error_list:
                if eline.strip().startswith("help:"):
                    help_flag = True
                if "|" in eline:
                    line_number = eline.split("|")[0].strip()
                    code_line = eline.split("|")[1].strip()
                    if line_number and code_line:
                        if help_flag:
                            suggestion_idx = line_number
                            suggestion_code = code_line
                        else:
                            error_idx = line_number
                            error_code = code_line
            if suggestion_idx and suggestion_code and error_idx and error_code:
                rust_code[int(error_idx) - 1] = rust_code[int(error_idx) - 1].replace(
                    error_code, suggestion_code
                )
                with open(file_path.replace(".rs", "_test.rs"), "w") as file:
                    file.write("".join(rust_code))
        subprocess.run(f"rm -rf {file_path.replace('.rs', '_test.rs')}", shell=True)
        subprocess.run(f"rm -rf {file_path.replace('.rs', '_test')}", shell=True)
        return return_code, no_errors

    def is_float(self, element: any) -> bool:
        if element is None:
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False

    def get_fn_args(self, code):
        arg_type_index = 0
        arg_name_index = 1
        arg_separator = " "
        fn_line = ""
        code_splitlines = code.splitlines()
        for i, line in enumerate(code_splitlines):
            if "f_gold ( " in line or "f_gold(" in line:
                fn_line = line.strip()
                break
        return_type = fn_line.split(" ")[0]
        if (
            "bool" in return_type
            or "long int" in return_type
            or "int" in return_type
            or "long long" in return_type
        ):
            return_type = "i32"
        elif "unsigned int" in return_type:
            return_type = "u32"
        elif "char" in return_type:
            return_type = "char"
        elif "float" in return_type:
            return_type = "f32"
        left_paren = fn_line.find("(")
        right_paren = fn_line.find(")")
        fn_name = "f_gold"
        fn_args = fn_line[left_paren + 1 : right_paren]

        array_idx = []
        args_list = fn_args.split(", ")
        for idx, arg in enumerate(args_list):
            if "[ ]" in arg or "[]" in arg:
                array_idx.append(idx)

        for i, arg in enumerate(args_list):
            if arg[0] == " ":
                arg = arg[1:]
                args_list[i] = arg
        fn_args_types = [
            (arg.split(arg_separator)[arg_type_index]).strip() for arg in args_list
        ]

        for i, t in enumerate(fn_args_types):
            if t == "unsigned":
                fn_args_types[i] = "unsigned int"
            elif t == "long":
                fn_args_types[i] = "int"
            elif t == "double":
                fn_args_types[i] = "float"
            elif not t:
                fn_args_types[i] = "int"
            if i in array_idx and t == "int":
                fn_args_types[i] = "int []"
            elif i in array_idx and t == "char":
                fn_args_types[i] = "char []"
            elif i in array_idx and t == "float":
                fn_args_types[i] = "float []"

        fn_args_names = [
            (arg.split(arg_separator)[arg_name_index]).strip()
            for arg in fn_args.split(",")
        ]

        str_args_names = ", ".join(fn_args_names)
        return fn_name, fn_args_types, str_args_names, return_type, fn_line

    def c_code_process(self, file_dir, package_name, file_name, f_filled, args_types):
        if "cpp" in file_dir:
            file_ext = ".cpp"
        else:
            file_ext = ".c"
        c_filepath = f"{file_dir}/{package_name}/{file_name}{file_ext}"
        with open(c_filepath, "r") as file:
            c_output = file.read()

        imports = """#include <stdio.h>
        #include <math.h>
        #include <stdlib.h>
        #include <limits.h>
        #include <stdbool.h>\n"""
        if "cpp" in file_dir:
            helper_funcs = """int min(int x, int y) { return (x < y)? x: y; }
            int max(int x, int y) { return (x > y)? x: y; }
            int cmpfunc (const void * a, const void * b) {return ( *(int*)a - *(int*)b );}
            int len (int arr [ ]) {return ((int) (sizeof (arr) / sizeof (arr)[0]));}
            void sort (int arr [ ], int n) {qsort (arr, n, sizeof(int), cmpfunc);}\n"""
        else:
            helper_funcs = ""

        main_line = 0
        c_lines = c_output.split("\n")
        for line_idx, line in enumerate(c_lines):
            if "main(" in line:
                main_line = line_idx
                continue
            if main_line and ("f_gold(param" in line or "f_gold(&param" in line or "f_gold(arr" in line):
                tokens = line.split(" ")
                function_call = "f_gold(0)"
                function_call_mutated = "f_gold(25)"
                array_declaration = ""
                for token in tokens:
                    if "f_gold" not in token:
                        continue
                    f_gold = token.replace("))", ")").replace("abs(", "")
                    function_call = f"{f_gold};"
                    char_list = ["'a'", "'b'", "'c'", "'d'", "'e'", "'f'", "'g'"]
                    str_list = ['"ab"', '"cb"', '"cf"', '"da"', '"ee"', '"fq"', '"gz"']
                    array_var_list = ["xv", "yq", "qe", "rp", "ww"]
                    mutation_target = "0"
                    # print(args_types)
                    for i, arg_type in enumerate(args_types):
                        param_str = f"param{i}[i]"
                        if (
                            arg_type == "int"
                            or arg_type == "unsigned int"
                            or arg_type == "float"
                        ):
                            function_call = function_call.replace(param_str, str(i + 1))
                            mutation_target = str(i + 1)

                        elif arg_type == "char":
                            function_call = function_call.replace(
                                param_str, char_list[i]
                            )
                        elif arg_type == "string":
                            function_call = function_call.replace(
                                param_str, str_list[i]
                            )
                        elif arg_type == "int []":
                            array_declaration += (
                                f"\tint {array_var_list[i]}[] = {{11,12}};\n"
                            )
                            function_call = function_call.replace(
                                param_str, array_var_list[i]
                            )
                        elif arg_type == "char []":
                            array_declaration += (
                                f"char {array_var_list[i]}[] = {{'a','b'}};\n"
                            )
                            function_call = function_call.replace(
                                param_str, array_var_list[i]
                            )
                        elif arg_type == "float []":
                            array_declaration += (
                                f"\tfloat {array_var_list[i]}[] = {{11,12}};\n"
                            )
                            function_call = function_call.replace(
                                param_str, array_var_list[i]
                            )

                    function_call = array_declaration + "\t" + function_call

                    function_call = function_call.replace("&", "").replace(
                        ".front()", ""
                    )

                    function_call_mutated = (
                        function_call.replace(mutation_target, "29")
                        .replace('"a",', '"c",')
                        .replace('"b"', '"d"')
                        .replace('"a"', '"d"')
                        .replace('"ab"', '"cd"')
                        .replace("'a'", "'c'")
                        .replace("'b'", "'d'")
                    )

        if "#include <bits/stdc++.h>" in c_output:
            c_output = c_output.replace("#include <bits/stdc++.h>", "")
            c_output = c_output.replace("int main() {", f"{f_filled}\nint main() {{")
            # print(c_output)
            with open(c_filepath, "w") as file:
                file.write(c_output)

        original = c_output
        target_lines = []
        is_block_comment = False
        main_line = 9999
        for i, line in enumerate(c_output.split("\n")):
            if "#include" in line or "# include" in line:
                target_lines.append(line)
                continue
            is_comment, is_block_comment = self.check_comment(
                line.strip(), is_block_comment
            )
            if "f_filled" in line:
                continue
            if " main(" in line:
                main_line = i
                continue
            if not is_comment and not is_block_comment and i < main_line:
                target_lines.append(line)

        c_output = "\n".join(target_lines)
        c_output = (
            c_output.replace("bool f_gold", "int f_gold")
            .replace("true", "1")
            .replace("false", "0")
            .replace("long int", "int")
            .replace("long int f_gold", "int f_gold")
            .replace("double f_gold", "float f_gold")
            .replace("string &", "string")
        )
        src_filepath = f"{file_dir}/{package_name}"
        c_filepath_processed = f"{src_filepath}/{file_name}_processed{file_ext}"
        c_filepath_wasm = f"{src_filepath}/{file_name}_towasm{file_ext}"
        c_filepath_wasm_mutated = f"{src_filepath}/{file_name}_towasm_mutated{file_ext}"

        with open(c_filepath_processed, "w") as file:
            processed_output = c_output.replace(helper_funcs, "")
            file.write(processed_output)

        with open(c_filepath_wasm, "w") as file:
            wasm_output = (
                imports
                + helper_funcs
                + c_output
                + "int main(void) {\n\t"
                + function_call
                + "\n}"
            )
            file.write(wasm_output)

        with open(c_filepath_wasm_mutated, "w") as file:
            wasm_output = (
                imports
                + helper_funcs
                + c_output
                + "int main(void) {\n\t"
                + function_call_mutated
                + "\n}"
            )
            file.write(wasm_output)

        subprocess.run(f"chmod -R a+rw {src_filepath}", shell=True)
        return processed_output, original
