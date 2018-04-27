using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM
{
    class SupportVectorMachine
    {
        //it's the sample of Support Vector Machine.  The functionality is not so completed yet. It is only for classifying 2 classes.
        //reference:
        //[0]:http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.4376&rep=rep1&type=pdf
        //[1]:https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec12.pdf
        //[2]:https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/tutorials/MIT6_034F10_tutor05.pdf
        //I use Gaussian Elimation method to solve the  matrix, though this is not the best method to solve linear equations, but it's easier to understand
        //referance:
        //[3]:https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
        // I found that Gaussian Elimination is not so good at dealing large Matrix (more than 20*20), and there is some problem on underdetermined linear system 
        //, so I survey to the simplified SMO
        //reference:
        //[4]:http://cs229.stanford.edu/materials/smo.pdf

        private double[][] INPUTVALUES;
        //2 dimensional array, can cantain several elements each rows,for instance, IRIS_data={sepal_length,sepal_width,petal_length,petal_width};
        private int[] LABELS;
        //shall specify +1 -1 corresponding to INPUTVALUES, it's as the output of f(x),to build SVM.
        /*       
        By Using Lagrange Multiplier Method,amd we simplify KKT conditions to get following Constraints:
        C0: w_vector=sumation(alpha_i*y_i*x_i)
        C1.sumation(alpha_i*y_i)=0
        C2. positive gutter function:y_i*(Trans_w_vector*x_i) sumation(alpha_i*y_i*kernel(x_i,x))+b=+1
        C3. negative gutter function: sumation(alpha_i*y_i*kernel(x_i,x))+b=-1
        C4. alpha_i>=0
        */
       // private double[] ALPHAs_b;//length of N+1 vector, structure is: alpha1,alpha2,...,alphaN,b
       // private double[][] ElementCoffS;//N+1xN+1 matrix
        /*structure of CoffS Matrix
         * y0   y1....  yN  0 
         * e00 e01.... e0N  1
         *e10  e11.....e1N  1
         *.
         * .
         * .
         * eN0 eN1....eNN  1
        */
        //private double[] VALofEQs;//length of N+1 
        //structure of VALofEQs vector: 0 1....1
        //CoffS*ALPHAs_b=VALofEQs->ALPHAs_b=VALofEQs*RevCoffS
        //ALPHAs_b vector is the result we want to evaluate.
       // private double[][] GAUSSMATRIX;//N+1*N+2 Matrix for Gaussian Elimination
        /*structure of GAUSS Matrix
         * y0   y1....  yN  0  0 
         * e00 e01.... e0N  1 y0
         *e10  e11.....e1N  1 y1
         * .
         * .
         * .
         * eN0 eN1.....eNN  1 yN
        */
        public double[] W_vector;
        /*
         * the normal vector  we shall evaluate to geneate classification function :
         * y=Trans(W_vector)(x_vector)+b
         * 
         * 
        */
        public double b;
        /*
         * the coefficient b is in this function:     y=Trans(W_vector)(x_vector)+b
         * b is the last element of ALPHAs_b
        */
        public SupportVectorMachine(double[][] TrainingData, int[] Traininglabels, KernelType tp)
        {
            if (TrainingData.Length > 0 && TrainingData.Length == Traininglabels.Length)
            {
                this.INPUTVALUES = TrainingData;
                this.LABELS = Traininglabels;
               
               
                SimplifiedSMO SSMO = new SimplifiedSMO(INPUTVALUES, LABELS,1e1, tp);



                // GAUSSMATRIX = BuildGuassianMatrix(tp);
                // ElementCoffS = BuildElementMatrix(tp);
                // StringBuilder a = new StringBuilder();
                //  for (int j = 0; j < GAUSSMATRIX[0].Length; j++)
                //  {
                //      a.Append("[" + GAUSSMATRIX[1][j].ToString() + "]");
                //  }
                //  Console.WriteLine(a.ToString());
                // ALPHAs_b = LeastSquareSolution(ElementCoffS, this.LABELS);
                // ALPHAs_b = GaussianElimination(GAUSSMATRIX);
                for (int i = 0; i <SSMO.ALPHAs.Length; i++)
                {
                    Console.WriteLine("alpha[{0}] is:{1}", i, SSMO.ALPHAs[i]);
                }
                W_vector = Wvector_Evaluate(INPUTVALUES,LABELS,SSMO.ALPHAs);
                b = SSMO.b;
            }
            else
            {
                if (TrainingData.Length <= 0)
                {
                    Console.WriteLine("Length of Traning Data is wrong");
                }
                if (Traininglabels.Length != TrainingData.Length)
                {
                    Console.WriteLine("data length of labels and inputvalues are not the same");
                }
            }
        }
        public SupportVectorMachine(double[] Previous_W_vector, double Previous_b)
        {
            this.W_vector = Previous_W_vector;
            this.b = Previous_b;
        }
        public int[] Predict(double[][] input_values)
        {
            int[] RESULTS = new int[input_values.Length];
            for (int i = 0; i < input_values.Length; i++)
            {
                double res = 0;
                for (int j = 0; j < input_values[i].Length; j++)
                {
                    res += W_vector[j] * input_values[i][j];
                }
                res += b;
                if (res >= 1)
                {
                    RESULTS[i] = 1;
                }
                else if (res <= -1)
                {
                    RESULTS[i] = -1;
                }
                else
                {
                    RESULTS[i] = 0;
                }
            }
            return RESULTS;
        }
        #region OBSOLETE-Matrix Evaluation
        /*
        private double[][] BuildElementMatrix(KernelType type)
        {

            double[][] CoffS = new double[INPUTVALUES.Length + 1][];
            CoffS[0] = new double[CoffS.Length];
            for (int i = 0; i < LABELS.Length; i++)
            {
                CoffS[0][i] = LABELS[i];
            }
            CoffS[0][CoffS.Length - 1] = 0;
            for (int i = 0; i < INPUTVALUES.Length; i++)
            {
                CoffS[i + 1] = new double[INPUTVALUES.Length + 1];
                for (int j = 0; j < INPUTVALUES.Length; j++)
                {
                    if (type == KernelType.Linear)
                    {
                        CoffS[i + 1][j] = LinearKernel(INPUTVALUES[i], INPUTVALUES[j]);
                    }
                    else if (type == KernelType.RBF)
                    {
                        CoffS[i + 1][j] = RBF_Kernel(INPUTVALUES[i], INPUTVALUES[j], 0.02);
                    }
                }
                CoffS[i + 1][INPUTVALUES.Length] = 1;
            }
            return CoffS;
        }

        private double[][] BuildGuassianMatrix(KernelType type)
        {
            double[][] GAUSSMatrix = new double[INPUTVALUES.Length + 1][];
            GAUSSMatrix[0] = new double[GAUSSMatrix.Length + 1];
            for (int i = 0; i < LABELS.Length; i++)
            {
                GAUSSMatrix[0][i] = LABELS[i];
            }
            GAUSSMatrix[0][GAUSSMatrix[0].Length - 2] = 0;
            GAUSSMatrix[0][GAUSSMatrix[0].Length - 1] = 0;
            for (int i = 0; i < INPUTVALUES.Length; i++)
            {
                GAUSSMatrix[i + 1] = new double[INPUTVALUES.Length + 2];
                for (int j = 0; j < INPUTVALUES.Length; j++)
                {
                    if (type == KernelType.Linear)
                    {
                        GAUSSMatrix[i + 1][j] = LinearKernel(INPUTVALUES[i], INPUTVALUES[j]);
                    }
                    else
                    {
                        GAUSSMatrix[i + 1][j] = RBF_Kernel(INPUTVALUES[i], INPUTVALUES[j], 0.02);
                    }
                }
                GAUSSMatrix[i + 1][INPUTVALUES.Length] = 1;
                GAUSSMatrix[i + 1][GAUSSMatrix.Length] = LABELS[i];
            }
            return GAUSSMatrix;
        }
        private static double LinearKernel(double[] u, double[] v)
        {
            double res = 0.0;
            if (u.Length == v.Length)
            {
                for (int i = 0; i < u.Length; i++)
                {
                    res += u[i] * v[i];
                }
            }
            return res;
        }
        private static double RBF_Kernel(double[] u, double[] v, double theta)
        {
            double res = 0.0;
            if (u.Length == v.Length)
            {
                for (int i = 0; i < u.Length; i++)
                {
                    res += (u[i] - v[i]) * (u[i] - v[i]);
                }
            }
            return Math.Pow(Math.E, -theta * res);
        }
        private static double Poly_Kernel(double[] u, double[] v, double times)
        {
            double res = 0.0;
            if (u.Length == v.Length)
            {
                for (int i = 0; i < u.Length; i++)
                {
                    res += u[i] * v[i];
                }
            }
            return Math.Pow(res, times);
        }
        
        private static double[] GaussianElimination(double[][] GAUSSMatrix)
        {
            double EPS = 1e-9;
            double[][] OperatingMatrix = GAUSSMatrix;
            int n = OperatingMatrix.Length;
            for (int i = 0; i < n; i++)
            {
                // Search for maximum in this column
                double MaxEL = Math.Abs(OperatingMatrix[i][i]);
                int maxRow = i;
                for (int k = i + 1; k < n; k++)
                {
                    if (MaxEL < Math.Abs(OperatingMatrix[k][i]))
                    {
                        MaxEL = Math.Abs(OperatingMatrix[k][i]);
                        maxRow = k;
                    }
                }
                // Swap maximum row with current row (column by column)
                for (int k = i; k < n + 1; k++)
                {
                    double temp = OperatingMatrix[maxRow][k];
                    OperatingMatrix[maxRow][k] = OperatingMatrix[i][k];
                    OperatingMatrix[i][k] = temp;
                }
                // Make all rows below this one 0 in current column
                for (int k = i + 1; k < n; k++)
                {
                    if (OperatingMatrix[i][i] == 0) {
                        Console.WriteLine("[{0}] is 0", i);
                        //OperatingMatrix[i][i] = 1e-7;
                    }
                    double c = -OperatingMatrix[k][i] / OperatingMatrix[i][i];
                    for (int j = i; j < n + 1; j++)
                    {
                        if (i == j)
                        {
                            OperatingMatrix[k][j] = 0;
                        }
                        else
                        {
                            OperatingMatrix[k][j] += c * OperatingMatrix[i][j];
                        }
                    }
                }
            }
            // Solve equation Ax=b for an upper triangular matrix A
            double[] res = new double[n];
            for (int i = n - 1; i >= 0; i--)
            {
                res[i] = OperatingMatrix[i][n] / OperatingMatrix[i][i];
                for (int k = i - 1; k >= 0; k--)
                {
                    OperatingMatrix[k][n] -= OperatingMatrix[k][i] * res[i];
                }
            }
            return res;
        }
        private static double[] LeastSquareSolution(double[][] ElementMatrix, int[] Labels)
        {
            double[][] Matrix = new double[ElementMatrix.Length][];
            double[] VLEQ = new double[ElementMatrix.Length];
            VLEQ[0] = 0;
            for (int i = 0; i < Labels.Length; i++)
            {
                VLEQ[i + 1] = Labels[i];
            }
            for (int i = 0; i < Matrix.Length; i++)
            {
                Matrix[i] = new double[ElementMatrix[i].Length];
                for (int j = 0; j < Matrix[i].Length; j++)
                {
                    for (int k = 0; k < Matrix.Length; k++)
                    {
                        Matrix[i][j] += ElementMatrix[i][k] * ElementMatrix[j][k];
                    }
                }
            }
            double[] temps = new double[VLEQ.Length];
            for (int i = 0; i < Matrix.Length; i++)
            {

                for (int j = 0; j < Matrix[i].Length; j++)
                {
                    for (int k = 0; k < VLEQ.Length; k++)
                    {
                        temps[i] += VLEQ[k] * Matrix[i][j];
                    }
                }
            }
            double[][] GaussianMatrix = new double[ElementMatrix.Length][];
            for (int i = 0; i < GaussianMatrix.Length; i++)
            {
                GaussianMatrix[i] = new double[ElementMatrix[i].Length + 1];
                for (int j = 0; j < GaussianMatrix[i].Length - 1; j++)
                {
                    GaussianMatrix[i][j] = Matrix[i][j];
                }
                GaussianMatrix[i][GaussianMatrix[i].Length - 1] = temps[i];
            }

            return GaussianElimination(GaussianMatrix);
        }*/
        /* private double[] Wvector_Evaluate()
              {
                  double[] W = new double[INPUTVALUES[0].Length];
                  for (int i = 0; i < INPUTVALUES.Length; i++)
                  {
                      for (int j = 0; j < INPUTVALUES[i].Length; j++)
                      {
                          W[j] += ALPHAs_b[i] * LABELS[i] * INPUTVALUES[i][j];
                          //by KTT C0: w_vector=sumation(alpha_i*y_i*x_i)
                      }
                  }
                  return W;
              }*/
        #endregion

        private double[] Wvector_Evaluate(double[][] input_values,int[]labels, double[] alphas)
        {
            double[] W = new double[input_values[0].Length];
            for (int i = 0; i < input_values.Length; i++)
            {
                for (int j = 0; j < input_values[i].Length; j++)
                {
                    W[j] += alphas[i] * labels[i] * input_values[i][j];
                    //by KKT C0: w_vector=sumation(alpha_i*y_i*x_i)
                }
            }
            return W;
        }
    }
    enum KernelType
    {
        Linear,
        RBF,
        Polynominal,
        InnerProduct
    }
    class SimplifiedSMO
    {
        public double[] ALPHAs;
        private double[][] INPUTVALUES;
        private int[] LABELS;
        public double b;
        private double tol = 1e-10;
        private int Max_Pass = 100;
        private double C = 1;
        KernelType Tp;
        public SimplifiedSMO(double[][] input_values, int[] labels,double C,KernelType Kerneltype)
        {
            this.INPUTVALUES = input_values;
            this.ALPHAs = new double[input_values.Length];
            Tp = Kerneltype;
            this.LABELS = labels;
            b = 0;
            int passes = 0;
            this.C = C;
            Random rdn = new Random();
            
            while (passes < Max_Pass)
            {
                int num_changed_alpha = 0;
                for (int i = 0; i < INPUTVALUES.Length; i++)
                {
                    double Ei =E_function(INPUTVALUES[i],i);
                    if ((Ei * LABELS[i] < -tol && ALPHAs[i] < C) || (Ei * LABELS[i] > tol && ALPHAs[i] > 0))
                    {
                        int j = 0;                        
                        j = rdn.Next(0, INPUTVALUES.Length - 1);
                        //Console.WriteLine(j);
                        while (j == i)j=rdn.Next(0, INPUTVALUES.Length - 1);
                        double Ej = E_function(INPUTVALUES[j],j);
                        //Console.WriteLine("Ej:{0}", Ej);
                        double ai = ALPHAs[i], aj = ALPHAs[j];
                        double L, H;
                        if (LABELS[i] != LABELS[j])
                        {
                             L = 0 > aj - ai ? 0 : aj - ai;
                             H = C < C + aj - ai ? C : C + aj - ai;
                        }
                        else
                        {
                            L = 0 > ai + aj - C ? 0 : ai + aj - C;
                            H = C < ai + aj ? C : ai + aj;
                        }
                        if (L == H)
                        {
                            //next i

                        }
                        else
                        {
                            double tau = 2 * InnerProduct(INPUTVALUES[i], INPUTVALUES[j]) - InnerProduct(INPUTVALUES[i], INPUTVALUES[i]) - InnerProduct(INPUTVALUES[j], INPUTVALUES[j]);
                            if (tau >= 0)
                            {
                                //next i
                            }
                            else
                            {
                                ALPHAs[j] = ALPHAs[j]-(E_function(INPUTVALUES[i],i) - E_function(INPUTVALUES[j], j)) *LABELS[j] / tau;
                                if (ALPHAs[j] > H) ALPHAs[j] = H;
                                else if (ALPHAs[j] < H && ALPHAs[j] > L) { }
                                else ALPHAs[j] = L;
                                if (Math.Abs(ALPHAs[j] - aj) < tol)
                                {
                                    //next i
                                }
                                else
                                {
                                     ALPHAs[i] = ALPHAs[i] + LABELS[i] * LABELS[j] * (aj - ALPHAs[j]);
                                    double b1 = b - E_function(INPUTVALUES[i], i) - LABELS[i] * ( ALPHAs[i]-ai) * InnerProduct(INPUTVALUES[i], INPUTVALUES[i]) - LABELS[j] * ( ALPHAs[j]-aj) * InnerProduct(INPUTVALUES[i], INPUTVALUES[j]);
                                    double b2 = b - E_function(INPUTVALUES[j], j) - LABELS[i] * (ALPHAs[i]-ai) * InnerProduct(INPUTVALUES[i], INPUTVALUES[j]) - LABELS[j] * (ALPHAs[j]-aj) * InnerProduct(INPUTVALUES[j], INPUTVALUES[j]);
                                    if (ALPHAs[i] > 0 && ALPHAs[i] < C) b = b1;
                                    else if (ALPHAs[j] > 0 && ALPHAs[j] < C) b = b2;
                                    else b = (b1 + b2) / 2;
                                    num_changed_alpha += 1;
                                }
                            }
                        }
                    }
                }
                if (num_changed_alpha == 0)
                {
                    passes += 1;
                   // Console.WriteLine("PASSES:{0}", passes);
                }
                else passes = 0;
            }
        }
        private static double InnerProduct(double[] u, double[] v)
        {
            double res = 0.0;
            for (int i = 0; i < u.Length; i++)
            {
                res += u[i] * v[i];
            }
            return res;
        }
        private static double LinearKernel(double[] u, double[] v)
        {
            return InnerProduct(u, v);
        }
        private static double RBF_Kernel(double[] u, double[] v,double Theta)
        {
            double res = 0.0;
            for (int i = 0; i < u.Length; i++)
            {
                res += (u[i] - v[i]) * (u[i] - v[i]);
            }
            return Math.Pow(Math.E, -res * Theta);
        }
        private  double E_function(double[]RowEle,int index_Label)
        {
            double res = 0.0;
            for (int i = 0; i < ALPHAs.Length; i++)
            {
                if (Tp == KernelType.InnerProduct||Tp==KernelType.Linear)
                {
                    res += ALPHAs[i] * LABELS[i] * LinearKernel(INPUTVALUES[i], RowEle);
                }
                else if (Tp == KernelType.RBF)
                {
                    res += ALPHAs[i] * LABELS[i] *RBF_Kernel(INPUTVALUES[i], RowEle,0.5);
                }
            }
            return res +b -LABELS[index_Label];
        }
    }
}
