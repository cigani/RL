#!/bin/bash

center=$1
action=$2

grep -A 7 "Grid Center index   : $center" checkfile | grep "Probability $action" |\
    awk '{print $4}'
