#include "safe_algorithm.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace fai {
namespace {

double clamp(double value, double low, double high) { return std::max(low, std::min(high, value)); }
enum class TokenKind { End, Number, Name, Symbol };
struct Token { TokenKind kind{TokenKind::End}; std::string text; double number{}; };

class Lexer {
public:
    explicit Lexer(std::string text) : text_(std::move(text)) { next(); }
    const Token& token() const { return token_; }
    void next() {
        while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_;
        if (position_ >= text_.size()) { token_ = {}; return; }
        const char c = text_[position_];
        if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && position_ + 1 < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_ + 1])))) {
            const size_t begin = position_++;
            while (position_ < text_.size() && (std::isdigit(static_cast<unsigned char>(text_[position_])) || text_[position_] == '.')) ++position_;
            if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
                ++position_; if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
                while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
            }
            token_ = {TokenKind::Number, text_.substr(begin, position_ - begin), 0}; token_.number = std::stod(token_.text); return;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const size_t begin = position_++;
            while (position_ < text_.size() && (std::isalnum(static_cast<unsigned char>(text_[position_])) || text_[position_] == '_')) ++position_;
            token_ = {TokenKind::Name, text_.substr(begin, position_ - begin), 0}; return;
        }
        std::string op(1, c); ++position_;
        if (position_ < text_.size()) { const std::string pair = op + text_[position_]; if (pair == "**" || pair == "<=" || pair == ">=" || pair == "==" || pair == "!=") { op = pair; ++position_; } }
        token_ = {TokenKind::Symbol, op, 0};
    }
    bool accept(const std::string& value) { if (token_.text != value) return false; next(); return true; }
    void expect(const std::string& value) { if (!accept(value)) throw std::runtime_error("expected '" + value + "'"); }
private:
    std::string text_; size_t position_{}; Token token_;
};

struct LogicalLine { int indent{}, line{}; std::string text; };
std::vector<LogicalLine> logicalLines(const std::string& source) {
    std::vector<LogicalLine> result; std::string pending; int pendingIndent=0,pendingLine=0,depth=0,lineNumber=0; size_t cursor=0;
    while(cursor<=source.size()) {
        ++lineNumber;const size_t end=source.find('\n',cursor);std::string raw=source.substr(cursor,end==std::string::npos?std::string::npos:end-cursor);
        if(!raw.empty()&&raw.back()=='\r')raw.pop_back();const size_t comment=raw.find('#');if(comment!=std::string::npos)raw.erase(comment);
        int indent=0;while(indent<int(raw.size())&&(raw[size_t(indent)]==' '||raw[size_t(indent)]=='\t'))indent+=raw[size_t(indent)]=='\t'?4:1;
        size_t first=raw.find_first_not_of(" \t");std::string content=first==std::string::npos?"":raw.substr(first);while(!content.empty()&&std::isspace(static_cast<unsigned char>(content.back())))content.pop_back();
        if(!content.empty()){if(pending.empty()){pendingIndent=indent;pendingLine=lineNumber;}if(!pending.empty())pending+=' ';pending+=content;for(char c:content){if(c=='(')++depth;else if(c==')')--depth;}if(depth<=0){result.push_back({pendingIndent,pendingLine,pending});pending.clear();depth=0;}}
        if(end==std::string::npos)break;cursor=end+1;
    }
    if(!pending.empty())result.push_back({pendingIndent,pendingLine,pending});return result;
}
} // namespace

struct Program::Expr {
    enum class Kind { Number, Name, Unary, Binary, Call, Conditional } kind{Kind::Number};
    double number{}; std::string text; std::vector<std::shared_ptr<Expr>> args;
};
struct Program::Statement {
    enum class Kind { Assign, If, Pass, Parameter } kind{Kind::Pass};
    int line{};std::string name;std::shared_ptr<Expr> expression;std::vector<Statement> body,otherwise;
};

namespace {
using Expr=Program::Expr;using ExprPtr=std::shared_ptr<Expr>;
ExprPtr makeExpr(Expr::Kind kind,std::string text={}){auto e=std::make_shared<Expr>();e->kind=kind;e->text=std::move(text);return e;}

class ExpressionParser {
public:
    explicit ExpressionParser(const std::string& source):lex_(source){}
    ExprPtr parse(){auto e=conditional();if(lex_.token().kind!=TokenKind::End)throw std::runtime_error("unexpected '"+lex_.token().text+"'");return e;}
private:
    Lexer lex_;
    ExprPtr conditional(){auto yes=disjunction();if(lex_.accept("if")){auto condition=disjunction();lex_.expect("else");auto no=conditional();auto e=makeExpr(Expr::Kind::Conditional);e->args={condition,yes,no};return e;}return yes;}
    ExprPtr disjunction(){auto e=conjunction();while(lex_.accept("or")){auto n=makeExpr(Expr::Kind::Binary,"or");n->args={e,conjunction()};e=n;}return e;}
    ExprPtr conjunction(){auto e=negation();while(lex_.accept("and")){auto n=makeExpr(Expr::Kind::Binary,"and");n->args={e,negation()};e=n;}return e;}
    ExprPtr negation(){if(lex_.accept("not")){auto e=makeExpr(Expr::Kind::Unary,"not");e->args={negation()};return e;}return comparison();}
    ExprPtr comparison(){auto e=addition();while(lex_.token().text=="<"||lex_.token().text=="<="||lex_.token().text==">"||lex_.token().text==">="||lex_.token().text=="=="||lex_.token().text=="!="){std::string op=lex_.token().text;lex_.next();auto n=makeExpr(Expr::Kind::Binary,op);n->args={e,addition()};e=n;}return e;}
    ExprPtr addition(){auto e=multiply();while(lex_.token().text=="+"||lex_.token().text=="-"){std::string op=lex_.token().text;lex_.next();auto n=makeExpr(Expr::Kind::Binary,op);n->args={e,multiply()};e=n;}return e;}
    ExprPtr multiply(){auto e=unary();while(lex_.token().text=="*"||lex_.token().text=="/"||lex_.token().text=="%"){std::string op=lex_.token().text;lex_.next();auto n=makeExpr(Expr::Kind::Binary,op);n->args={e,unary()};e=n;}return e;}
    ExprPtr unary(){if(lex_.token().text=="+"||lex_.token().text=="-"){std::string op=lex_.token().text;lex_.next();auto e=makeExpr(Expr::Kind::Unary,op);e->args={unary()};return e;}return power();}
    ExprPtr power(){auto e=primary();if(lex_.accept("**")){auto n=makeExpr(Expr::Kind::Binary,"**");n->args={e,unary()};return n;}return e;}
    ExprPtr primary(){
        if(lex_.token().kind==TokenKind::Number){auto e=makeExpr(Expr::Kind::Number);e->number=lex_.token().number;lex_.next();return e;}
        if(lex_.token().kind==TokenKind::Name){std::string name=lex_.token().text;lex_.next();if(!lex_.accept("(")){if(name=="True"||name=="False"){auto e=makeExpr(Expr::Kind::Number);e->number=name=="True";return e;}return makeExpr(Expr::Kind::Name,name);}auto e=makeExpr(Expr::Kind::Call,name);if(!lex_.accept(")")){do{e->args.push_back(conditional());}while(lex_.accept(","));lex_.expect(")");}return e;}
        if(lex_.accept("(")){auto e=conditional();lex_.expect(")");return e;}throw std::runtime_error("expression expected");
    }
};

size_t assignmentPosition(const std::string& text){int depth=0;for(size_t i=0;i<text.size();++i){if(text[i]=='(')++depth;else if(text[i]==')')--depth;else if(text[i]=='='&&depth==0){char before=i?text[i-1]:'\0',after=i+1<text.size()?text[i+1]:'\0';if(before!='<'&&before!='>'&&before!='!'&&before!='='&&after!='=')return i;}}return std::string::npos;}

std::vector<Program::Statement> parseBlock(const std::vector<LogicalLine>&lines,size_t&index,int indent,std::map<std::string,Parameter>&parameters){
    std::vector<Program::Statement>out;
    while(index<lines.size()){
        const auto&line=lines[index];if(line.indent<indent)break;if(line.indent>indent)throw std::runtime_error("Line "+std::to_string(line.line)+": unexpected indentation");if(line.text=="else:")break;
        Program::Statement statement;statement.line=line.line;
        if(line.text=="pass"){++index;out.push_back(std::move(statement));continue;}
        if(line.text.rfind("if ",0)==0&&line.text.back()==':'){statement.kind=Program::Statement::Kind::If;statement.expression=ExpressionParser(line.text.substr(3,line.text.size()-4)).parse();++index;if(index>=lines.size()||lines[index].indent<=indent)throw std::runtime_error("Line "+std::to_string(line.line)+": if needs an indented body");statement.body=parseBlock(lines,index,lines[index].indent,parameters);if(index<lines.size()&&lines[index].indent==indent&&lines[index].text=="else:"){++index;if(index>=lines.size()||lines[index].indent<=indent)throw std::runtime_error("Line "+std::to_string(line.line)+": else needs an indented body");statement.otherwise=parseBlock(lines,index,lines[index].indent,parameters);}out.push_back(std::move(statement));continue;}
        size_t eq=assignmentPosition(line.text);if(eq==std::string::npos)throw std::runtime_error("Line "+std::to_string(line.line)+": statement is not allowed");statement.name=line.text.substr(0,eq);while(!statement.name.empty()&&std::isspace(static_cast<unsigned char>(statement.name.back())))statement.name.pop_back();size_t first=statement.name.find_first_not_of(" \t");statement.name=first==std::string::npos?"":statement.name.substr(first);if(statement.name.empty()||(!std::isalpha(static_cast<unsigned char>(statement.name[0]))&&statement.name[0]!='_'))throw std::runtime_error("Line "+std::to_string(line.line)+": invalid variable name");
        statement.expression=ExpressionParser(line.text.substr(eq+1)).parse();statement.kind=Program::Statement::Kind::Assign;
        if(statement.expression->kind==Expr::Kind::Call&&statement.expression->text=="parameter"){if(statement.expression->args.size()!=3)throw std::runtime_error("Line "+std::to_string(line.line)+": parameter needs default, min, max");for(const auto&a:statement.expression->args)if(a->kind!=Expr::Kind::Number)throw std::runtime_error("Line "+std::to_string(line.line)+": parameter values must be numbers");Parameter p{statement.expression->args[0]->number,statement.expression->args[1]->number,statement.expression->args[2]->number};if(p.low>p.defaultValue||p.defaultValue>p.high)throw std::runtime_error("Line "+std::to_string(line.line)+": invalid parameter range");parameters[statement.name]=p;statement.kind=Program::Statement::Kind::Parameter;}
        ++index;out.push_back(std::move(statement));
    }return out;
}

double evaluate(const ExprPtr&e,const std::unordered_map<std::string,double>&env){
    if(!e)return 0;switch(e->kind){
        case Expr::Kind::Number:return e->number;
        case Expr::Kind::Name:{auto it=env.find(e->text);return it==env.end()?0.0:it->second;}
        case Expr::Kind::Unary:{double a=evaluate(e->args[0],env);if(e->text=="-")return-a;if(e->text=="not")return !bool(a);return a;}
        case Expr::Kind::Conditional:return evaluate(e->args[bool(evaluate(e->args[0],env))?1:2],env);
        case Expr::Kind::Call:{std::vector<double>a;for(const auto&v:e->args)a.push_back(evaluate(v,env));if(e->text=="abs")return a.empty()?0:std::abs(a[0]);if(e->text=="sign")return a.empty()?0:(a[0]>0)-(a[0]<0);if(e->text=="sqrt")return a.empty()?0:std::sqrt(std::max(0.0,a[0]));if(e->text=="min")return a.empty()?0:*std::min_element(a.begin(),a.end());if(e->text=="max")return a.empty()?0:*std::max_element(a.begin(),a.end());if(e->text=="clamp")return a.size()<3?0:clamp(a[0],a[1],a[2]);return 0;}
        case Expr::Kind::Binary:{const std::string&op=e->text;double a=evaluate(e->args[0],env);if(op=="and"&&!bool(a))return 0;if(op=="or"&&bool(a))return 1;double b=evaluate(e->args[1],env);if(op=="+")return a+b;if(op=="-")return a-b;if(op=="*")return a*b;if(op=="/")return std::abs(b)>1e-9?a/b:0;if(op=="%")return std::abs(b)>1e-9?std::fmod(a,b):0;if(op=="**")return std::pow(a,clamp(b,-4.0,4.0));if(op=="<")return a<b;if(op=="<=")return a<=b;if(op==">")return a>b;if(op==">=")return a>=b;if(op=="==")return a==b;if(op=="!=")return a!=b;if(op=="and")return bool(a)&&bool(b);if(op=="or")return bool(a)||bool(b);return 0;}
    }return 0;
}
void execute(const std::vector<Program::Statement>&statements,std::unordered_map<std::string,double>&env){for(const auto&s:statements){if(s.kind==Program::Statement::Kind::Assign)env[s.name]=evaluate(s.expression,env);else if(s.kind==Program::Statement::Kind::If)execute(bool(evaluate(s.expression,env))?s.body:s.otherwise,env);}}
} // namespace

Program::Program():statements_(std::make_shared<std::vector<Statement>>()){}Program::Program(const std::string&source):Program(){compile(source);}
bool Program::compile(const std::string&source){valid_=false;error_.clear();statements_=std::make_shared<std::vector<Statement>>();parameters_.clear();try{auto lines=logicalLines(source);size_t index=0;if(!lines.empty()){*statements_=parseBlock(lines,index,lines.front().indent,parameters_);if(index!=lines.size())throw std::runtime_error("Line "+std::to_string(lines[index].line)+": unexpected else or indentation");}valid_=true;}catch(const std::exception&e){error_=e.what();statements_=std::make_shared<std::vector<Statement>>();parameters_.clear();}return valid_;}
Outputs Program::run(const std::unordered_map<std::string,double>&inputs,const std::map<std::string,double>&values)const{std::unordered_map<std::string,double>env=inputs;env["brake"]=0;env["overtake"]=0;env["recharge"]=0;env["pit_request"]=0;env["pit_tyre"]=1;for(const auto&[name,p]:parameters_){auto it=values.find(name);env[name]=clamp(it==values.end()?p.defaultValue:it->second,p.low,p.high);}if(valid_&&statements_)execute(*statements_,env);auto get=[&](const char*name){auto i=env.find(name);return i==env.end()?0.0:i->second;};Outputs o;o.steering=clamp(get("steering"),-1.0,1.0);o.throttle=clamp(get("throttle"),0.0,1.0);o.brake=clamp(get("brake"),0.0,1.0);o.overtake=clamp(get("overtake"),0.0,1.0);o.recharge=clamp(get("recharge"),0.0,1.0);o.pitRequest=clamp(get("pit_request"),0.0,1.0);o.pitTyre=int(std::round(clamp(get("pit_tyre"),0.0,3.0)));return o;}

} // namespace fai
