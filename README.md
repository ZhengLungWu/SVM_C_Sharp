# SVM_C_Sharp
>1.something balabala...

it is the basic SVM classifier written in C#. 
Although there are some good tool(like SVMLIB), but the C/C++ code is too difficult for me to understand.
If I use those packages, I feel like I just utilize black boxes to do alien job. 

And in my working field(manufacturing/heavy industry),python is not so suitable to cooperate with PLC/Industrial control system/machine tools/plenty of sensors....balabala... 

Therefore I decided to build a SVM using C# by my own. 

I surveyed the internet and learned how to make the SVM work. I firstly used the method revealed in 1979, and it needs plenty of QR calculation, and the matrix evaluation made me crazy. So I turned to SMO algorithm. That is a genuis algorithm, which makes SVM highy efficient and fast. I use Simplified SMO as solver.

In this stage, this SVM can only classify 2 classes. it will extend to multi class SVM soon.

## How to use it  
>2.Training

>>2.1input  

the type:
``` csharp 
doule[][]inputvalues; //for input data, 
int[]labels;// for labels
```

Take iris data for example:

the data length is 150, and it has 4 elements each row.


``` csharp
doule[][]IRIS_data=new double[150][];
for(int i=0;i<150;i++)
{
IRIS_data[i]={parameter1,parameter2,parameter3,parameter4};// not legeal C# syntax, but for expressing.
}

int[]labels 
```

int[]labels is the array that contains +1/-1, to let machine distingulish the class of corresponding row.

for example: there are 2 kinds of iris

type A:+1, type B:-1

so the array labels will be like: 

```csharp
labels={1,-1,1....};// not legeal C# syntax, but for expressing.
```
>>2.2output

the output type :
```csharp
double[]W_vector;// for W vector, 
double b;//b coefficient
```
because the target function we want is: 
```csharp
double y=0;
for(int i=0;i<W_vector.Length;i++)y+=(W_vector[i])*(X[i])+b;
```
, use W and b to refer the result of new input data y >=+1 or y<=-1

>3.predict

the data to be predicted is 
```csharp
double[][]newdata
```
and the output will be
```csharp
int[]Result_Labels
```
4.reference:

[0]http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.4376&rep=rep1&type=pdf

[1]https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec12.pdf

[2]https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/tutorials/MIT6_034F10_tutor05.pdf

[3]https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/

[4]http://cs229.stanford.edu/materials/smo.pdf
