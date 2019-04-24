#include <stdio.h>
#include <string.h>

struct XNode {
    int val;
    struct XNode *next;
};


static void xms_merge_sort() {

}


static void xms_partition(struct XNode *head, struct XNode **front, struct XNode **back) {

}


static void pre_compute_lps_arr(int lps_arr[], const char *key, int len) {
    lps_arr[0] = 0;
    int tl = 0;
    int i = 1;
    while (i < len) {
        if (key[i] == key[tl]) {
            lps_arr[i++] = ++tl;
        }
        else if (tl > 0) {
            tl = lps_arr[tl - 1];
        } else {
            lps_arr[i++] = 0;
        }
    }
}


static void kmp_match(const char *key, const char *str) {
    int m = strlen(key);
    int n = strlen(str);
    int lps_arr[m];
    int count = 0;
    pre_compute_lps_arr(lps_arr, key, m);
    int i = 0, j = 0;
    while (i < n) {
        if (key[j] == str[i]) {
            j++;
            i++;
        }
        else if (j == m) {
            printf("matched key at loc: %d\n", i - j);
            j = lps_arr[j - 1];
            count++;
        } else {
            if (j > 0) {
                j = lps_arr[j - 1];
            } else {
                i++;
            }
        }
    }
    printf("matched %d times !\n", count);
}

int main() {
    kmp_match("aaacaaaa", "asjkjdaaacaaaaopolaaacaaaaoo");
    return 0;
}