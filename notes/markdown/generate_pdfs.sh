pandoc 0_introduction.md -o ../0_introduction.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 1_background.md -o ../1_background.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 2_intro_to_rl.md -o ../2_intro_to_rl.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 3_value_functions.md -o ../3_value_functions.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 4_dqn_extensions.md -o ../4_dqn_extensions.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 5_policy_gradients.md -o ../5_policy_gradients.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 6_alphago.md -o ../6_alphago.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 7_practical_concerns.md -o ../7_practical_concerns.pdf -V geometry:margin=1in --variable urlcolor=cyan

pandoc 8_sota.md -o ../8_sota.pdf -V geometry:margin=1in --variable urlcolor=cyan
