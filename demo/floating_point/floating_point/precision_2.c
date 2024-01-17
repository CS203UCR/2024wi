#include <stdio.h>

int main(int argc, char **argv)
{
    float a, b, c;
    a = 1280.245;
    b = 0.0004;
    c = (a + b)*10.0;
    printf("(1280.245 + 0.0004)*10 = %f\n",c);
    c = a*10.0 + b*10.0;
    printf("1280.245*10 + 0.0004*10 = %f\n",c);
    return 0;
}