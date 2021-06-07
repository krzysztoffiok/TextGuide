# TextGuide
A clever text truncation method.

Example use case - long text classification. If you're using a transformer model with a standard token limit of 512 and your text instnaces are longer than that, Text Guide can help you
to improve prediction performance without employing computationally expensive very large models with greater token limits. 

This code was published to accompany the <a href=https://arxiv.org/pdf/2104.07225.pdf>Text Guide</a> paper which intruduces the method.

Text Guide allows for high quality and computationally nonexpensive text truncation. It selects the important parts of the original text instances based on feature importance obtained from a machine learning classifier (for example gradient boosting) trained earlier on features from a simple model based on counting occurrences of tokens, for example a bag of words model.

Feel free to use the code in your works. If you do, the only thing we kindly ask in return is to cite our research: </br>
Fiok, K., Karwowski, W., Gutierrez, E., Davahli, M. R., Wilamowski, M., Ahram, T., ... & Zurada, J. (2021). Text Guide: Improving the quality of long text classification by a text selection method based on feature importance. arXiv preprint arXiv:2104.07225.

How to use the code:
1) Copy this repository
2) Run the example simply by typing: python3 run.py
3) The Python3 packages that you might need to install before using Text Guide are: feather, dask and dask[dataframe]
4) If you wish to use Text Guide for your own data, you need to edit the ./configs/your_config_file.py . You'll also need to provide a python dictionary with keys being tokens deemed as important and values being feature importances. If you want to play, you can write the dictionary withouth training any specific models, just use your own intelligence to define important tokens and corresponding importances.
