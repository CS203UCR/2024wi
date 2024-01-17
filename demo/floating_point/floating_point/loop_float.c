#include <stdio.h>

int main(int argc, char **argv)
{
    float i=0.0;
    while(i < 33554432) i++;
    printf("We're done! %f\n",i);
    return 0;
}