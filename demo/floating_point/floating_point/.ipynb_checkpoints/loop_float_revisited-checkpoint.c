#include <stdio.h>

union Number
{
    float f;
    unsigned char str[8]; 
} number;

int main(int argc, char **argv)
{
    float i=1.0;
    int k;
    while(i > 0) i++;
    printf("We're done! %f\n",i);
    number.f = i;
    for(k=0;k<8;k++)
        printf("%x\n",number.str[k]);
    printf("\n");
    
    return 0;
}