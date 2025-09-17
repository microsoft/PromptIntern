import traceback
import re

def eval_python(code, testcase):
    local_var = {}
    exit_code = None
    try:
        function = code.split("python```")[1].split("```")[0] if "```" in code else code.strip(".")
        function = function + "\n" + testcase
        print(function)
        import_lib = [i for i in function.split("\n") if i.startswith("import ") or i.startswith("from ")]
        LIBVAR = locals()
        exec("\n".join(import_lib), globals(), LIBVAR)
        GLOBALVAR = globals()
        GLOBALVAR.update(LIBVAR)

        exec("\n".join([function]), GLOBALVAR, local_var)
        exit_code = 0
        return {"exit_code": exit_code,"input": testcase,"output":local_var["result"]}
    except KeyError as e:
        if str(e) == "'result'":
            exit_code = 0
            return {"exit_code": exit_code,"input": testcase,"output":'<The test case does not return an output>'}
    except:
        exc_info=traceback.format_exc()
        error_line = None
        if "File \"<string>\"," in exc_info:
            pattern = r'File "<string>", line (\d+)'  
            match = re.search(pattern, exc_info)  
            
            if match:  
                line_number = match.group(1)  
                error_line=function.split("\n")[int(line_number)-1]

        exit_code = 1
        return {"exit_code": exit_code,"input": testcase, 'output': None, "error":exc_info.strip().splitlines()[-1], "error_line": error_line}

if __name__ == "__main__":
    code = '''
python```
def sort_matrix(M):\r\n    result = sorted(M, key=sum)\r\n    return result\n
```
'''
    testcase = '''
assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]
assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]
assert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]
'''

    print(eval_python(code,testcase))