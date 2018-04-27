# SVM_C_Sharp
something balabala...
it is the basic SVM classifier written in C#. 
Although there are some good tool(like SVMLIB), but the C/C++ code is too difficult for me to understand.
If I use those packages, I feel like I just utilize black boxes to do alien job. 
And in my working field(manufacturing/heavy industry),python is not so suitable to cooperate with PLC/Industrial control system/machine tools/plenty of sensors....balabala... 
Therefore I decided to build a SVM using C# by my own. 
I surveyed the internet and learned how to make the SVM work. I firstly used the method revealed in 1979, and it needs plenty of QR calculation, and the matrix evaluation made me crazy. So I turned to SMO algorithm. That is a genuis algorithm, which makes SVM highy efficient and fast. I use Simplified SMO as solver.
In this stage, this SVM can only classify 2 classes. it will extend to multi class SVM soon.

  
how to use it
input  
the type: doule[][] for input data, int[] for labels

Take iris data for example:
the data length is 150, and it has 4 elements each row.
doule[][]IRIS_data=new double[150][];
for i form 0-149
IRIS_data[i]={parameter1,parameter2,parameter3,parameter4};

int[]labels is the array that contains +1/-1, to let machine distingulish the class of corresponding row.
for example: there are 2 kinds of iris
type A:+1, type B:-1
so the array labels will be like: {1,-1,1....};

output
the output type : double[] for W vector, double for b coefficient
because the target function we want is: yi=sumation(Wi)(Xi)+b, use W and b to refer the result of new input dat ynew >=+1 or <=-1
predict
 the data to be predicted is double[][]newdata
 and the output will be int[]Result_Labels
