 all: hessian.exe

 hessian.exe: calc_hess.o Hessian.h 
	g++ -o hessian.exe calc_hess.o Hessian.h 

calc_hess.o: Hessian.h 
	g++ -c calc_hess.cpp

clean:
	rm calc_hess.o hessian.exe 