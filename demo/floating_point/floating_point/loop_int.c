#include <stdio.h>

int main(int argc, char **argv)
{
    int i=0;
    while(i < 33554432) i++;
    printf("We're done! %d\n", i);
    return 0;
}