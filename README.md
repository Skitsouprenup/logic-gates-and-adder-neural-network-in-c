# logic-gates-and-adder-neural-network
A simple neural network model for [logic gates](https://en.wikipedia.org/wiki/Logic_gate) and [logic gates](https://en.wikipedia.org/wiki/Adder_(electronics)) in C programming language. This model is built for educational purposes only and not for production. No third-party AI library is used in this project because the reason I created this project is to understand how neural network works.

I have a quick demo of this project in this [link](https://youtu.be/71N2ihtNK80)

# Testing this project
I tested this project in linux with gcc compiler. To compile this project using gcc, type this command:  
To compile adder2.c -> `gcc -o adder2 adder2.c -lm`  
To compile gates.c -> `gcc -o gates gates.c -lm`

After compiling, execute the compiled file. In linux terminal, point the terminal to the folder where the executables are located and type this:  
To run 'adder2' executable file -> `./adder2 f`  
The 'f' character is a flag where the program will use finite difference as cost reduction method. If the character is 'b', the program will use back propagation. If the character is not 'f' or 'b', back propagation will be used by default.

To run 'gates' executable file -> `./gates`