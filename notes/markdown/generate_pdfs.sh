pandoc 1_statistical_background.md -o ../1_statistical_background.pdf -V geometry:margin=1in --variable urlcolor=cyan

echo '1 done'

pandoc 2_intro_to_rl.md -o ../2_intro_to_rl.pdf -V geometry:margin=1in --variable urlcolor=cyan

echo '2 done'

pandoc 3_value_functions.md -o ../3_value_functions.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 4_dqn_extensions.md -o ../4_dqn_extensions.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 5_policy_gradients.md -o ../5_policy_gradients.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 6_alphago.md -o ../6_alphago.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 7_practical_concerns.md -o ../7_practical_concerns.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 8_sota.md -o ../8_sota.pdf -V geometry:margin=1in --variable urlcolor=cyan
