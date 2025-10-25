#include <iostream>
#include <string>
#include <list>
#include <stack>
#include <sstream>
//using namespace std;


std::string extractSubstring(const std::string& text,const std::string& start_key,const std::string& end_key){
    size_t start_pos = text.find(start_key);
    if (start_pos == std::string::npos){
            return "";
    }
    size_t end_pos = text.find(end_key);
    if (end_pos == std::string::npos){
        return "";
}
    start_pos += start_key.length();
    
    return text.substr(start_pos,end_pos-start_pos);
    

}
double getNumbersFromCalc(std::string answer){
    
    std::stack<double> numbers;  // Стек для чисел
    std::stack<char> operators;   // Стек для операторов

    std::istringstream iss(answer);
    std::string token;
    while (iss >> token){

        if (isdigit(token[0])) {  // Если это число
            numbers.push(std::stod(token));

    }else{
        while (!operators.empty() && 
            (operators.top() == '*' || operators.top() == '/') && 
            (token[0] == '+' || token[0] == '-')) {
             double b = numbers.top(); numbers.pop();
            double a = numbers.top(); numbers.pop();
            char op = operators.top(); operators.pop();
            switch (op) {
                case '+': numbers.push(a + b); break;
                case '-': numbers.push(a - b); break;
                case '*': numbers.push(a * b); break;
                case '/': numbers.push(a / b); break;
            }

    }
    operators.push(token[0]); 
        }
}
    while (!operators.empty()) {
        double b = numbers.top(); numbers.pop();
        double a = numbers.top(); numbers.pop();
        char op = operators.top(); operators.pop();

        switch (op) {
            case '+': numbers.push(a + b); break;
            case '-': numbers.push(a - b); break;
            case '*': numbers.push(a * b); break;
            case '/': numbers.push(a / b); break;
        }
    }

    return numbers.top(); 
}


int main(int argc, char* argv[]){

    if (argc < 2 || argv[1] == nullptr) {
        std::cerr << "Ошибка: нужно передать строковый аргумент!" << std::endl;
        return 1;
    }
    std::string answer = argv[1];
    
    std::cout << getNumbersFromCalc(extractSubstring(answer,"/calc","/calcstop"))<<std::endl;

    return 0;
}