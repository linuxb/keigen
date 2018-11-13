//
// Created by nuxeslin on 2018/9/20.
//
#include <iostream>
#include <vector>
#include <map>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <Eigen/Dense>

using namespace std;
using namespace llvm;

enum Token_Type {
    EOF_TOKEN = 0,
    NUMERIC_TOKEN,
    IDENTIFIER_TOKEN,
    DEF_TOKEN
};

static map<char, int> operators_precedence;

FILE *src_file;
static string identifier_name;
static int tensor_val;
static int current_token;

// ast
class BaseHLO {
public:
    virtual ~BaseHLO() {};
};

static void init_operators_precedence() {
    operators_precedence['+'] = 1;
    operators_precedence['-'] = 1;
    operators_precedence['*'] = 2;
    operators_precedence['/'] = 2;
}

class VariableHLO : public BaseHLO {
public:
    inline explicit VariableHLO(string &varname) : varname_(varname) {}

private:
    string varname_;
};

class NumericHLO : public BaseHLO {
public:
    inline explicit NumericHLO(int tnval) : tn_val_(tnval) {}

private:
    int tn_val_;
};

class BinaryHLO : public BaseHLO {
public:
    BinaryHLO(const string &op, BaseHLO *lhs, BaseHLO *rhs) :
            bin_operator_(op), lhs_(lhs), rhs_(rhs) {}

private:
    string bin_operator_;
    BaseHLO *lhs_, *rhs_;
};

class FunctionDeclHLO {
public:
    FunctionDeclHLO(const string &func_name, const vector<string> &args) :
            func_name_(func_name), args_(args) {}

private:
    string func_name_;
    vector<string> args_;
};

class FunctionDefHLO {
public:
    FunctionDefHLO(FunctionDeclHLO *proto, BaseHLO *kernel) :
            proto_(proto), kernel_(kernel) {}
private:
    FunctionDeclHLO *proto_;
    BaseHLO *kernel_;
};

class FunctionCallHLO : public BaseHLO {
public:
    FunctionCallHLO(const string &callee, vector<BaseHLO*> &args) :
            callee_(callee), args_(args) {}

private:
    string callee_;
    vector<BaseHLO*> args_;
};

static int get_token() {
    static int last_char = ' ';
    while (isspace(last_char)) last_char = fgetc(src_file);
    if (isalpha(last_char)) {
        identifier_name = (char)last_char;
        while (isalnum(last_char = fgetc(src_file))) {
            identifier_name += (char)last_char;
        }

        if ("def" == identifier_name) return DEF_TOKEN;
        return IDENTIFIER_TOKEN;
    }

    if (isdigit(last_char)) {
        string str_num;
        do {
            str_num += (char)last_char;
            last_char = fgetc(src_file);
        } while (isdigit(last_char));
        tensor_val = (int)strtod(str_num.c_str(), nullptr);
        return NUMERIC_TOKEN;
    }
    // annotation
    if ('#' == last_char) {
        do {
            last_char = (char)fgetc(src_file);
        } while (last_char != EOF && last_char != '\n' && last_char != '\r');
        // continue
        if (last_char != EOF) return get_token();
    }

    //end of file
    if (EOF == last_char) return EOF_TOKEN;
    int this_char = last_char;
    last_char = fgetc(src_file);
    return this_char;
}

static void next_token() {
    current_token = get_token();
}

static int get_operator_precedence() {
    if (!isascii(current_token))
        return -1;
    int tok_prec = operators_precedence[current_token];
    if (tok_prec <= 0)
        return -1;
    return tok_prec;
}

//parser definitions
#define __handle_err__(op_name) \
    printf("syntax error when parse ##op_name\n");

#define __parser_err__(op_name) \
    do { \
         __handle_err__(#op_name); \
        return nullptr; \
    } while(0);

#define PARSER_DECL(name) \
    static BaseHLO* name##_parser();

PARSER_DECL(base)
PARSER_DECL(numeric)
PARSER_DECL(identifier)
PARSER_DECL(binary_op)
PARSER_DECL(expression)
PARSER_DECL(paran)

static BaseHLO* base_parser() {
    switch (current_token) {
        default: {
            __handle_err__(unresolved_identifier) //NOLINT
            return nullptr;
        }
        case IDENTIFIER_TOKEN: return identifier_parser();
        case NUMERIC_TOKEN: return numeric_parser();
        case '(': return paran_parser();
    }
}

static BaseHLO* numeric_parser() {
    BaseHLO *node = new NumericHLO(tensor_val);
    next_token();
    return node;
}

static BaseHLO* identifier_parser() {
    string id_name = identifier_name;
    next_token();
    // if launch a opkernel
    if (current_token != '(') return new VariableHLO(id_name);
    // parse args
    next_token();
    vector<BaseHLO*> args;
    if (current_token != ')') {
        // parser args (expr1, expr2, ...)
        while (true) {
            BaseHLO *arg = expression_parser();
            // err
            if (!arg) {
                __handle_err__(identifier) // NOLINT
                return nullptr;
            }
            args.push_back(arg);
            if (current_token == ')') break;
            if (current_token != ',') {
                __handle_err__(identifier) // NOLINT
                return nullptr;
            }
            next_token();
        }
    }
    next_token();
    return new FunctionCallHLO(id_name, args);
}

static FunctionDeclHLO* func_decl_parser() {
    if (current_token != DEF_TOKEN) {
        __handle_err__(func_decl) //NOLINT
        return nullptr;
    }
    string func_name = identifier_name;
    next_token();
    if (current_token != '(') __parser_err__(func_decl) //NOLINT
    next_token();
    vector<string> args_names;
    while (true) {
        if (current_token != IDENTIFIER_TOKEN) __parser_err__(func_decl) //NOLINT
        args_names.push_back(identifier_name);
        next_token();
        if (current_token == ')') break;
        if (current_token != ',') __parser_err__(func_decl) //NOLINT
        next_token();
    }
    next_token();
    return new FunctionDeclHLO(func_name, args_names);
}

static BaseHLO* binary_op_parser(int prev_prec, BaseHLO *lhs) {
    while (true) {
        int cur_prec = get_operator_precedence();
        if (cur_prec <= prev_prec)
            return lhs;
        int bin_op = current_token;
        next_token();
        BaseHLO *rhs = base_parser();
        if (!rhs) return nullptr;
        int next_prec = get_operator_precedence();
        if (cur_prec < next_prec) {
            rhs = binary_op_parser(cur_prec, rhs);
            if (!rhs) return nullptr;
        }
        lhs = new BinaryHLO(std::to_string(bin_op), lhs, rhs);
    }
}

static BaseHLO* expression_parser() {
    BaseHLO *lhs = base_parser();
    if (!lhs) return nullptr;
    return binary_op_parser(0, lhs);
}

static BaseHLO* paran_parser() {
    next_token();
    BaseHLO *expr = base_parser();
    if (!expr) return nullptr;
    if (current_token != ')')
        return nullptr;
    return expr;
}

void InitMat(float *mat, int row_num, int col_num) {
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            mat[i * col_num + j] = 1.0;
        }
    }
}

int main() {
    auto thr_id = llvm::get_threadid();
    printf("current thread: %llu\n", thr_id);
    static LLVMContext my_ctx;
    auto module_ob = new Module("lollipop", my_ctx);
    auto rref = std::move(6);
    Eigen::Matrix2d mat_z(2, 2);
    mat_z(0, 0) = 6;
    cout << "res matrix: " << mat_z << endl;
    float xmat[6 * 6];
    InitMat(xmat, 6, 6);
    cout << xmat[0] << endl;
    return 0;
}

