#include <stdio.h>

int main(int argc, char **argv)
{
    int count;
    for(count = 0 ; count < 10000000; count++);
    printf("%d\n",count);
    return 0;
}