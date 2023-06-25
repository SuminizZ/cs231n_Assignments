import numpy as np

from ..rnn_layers import *

class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=np.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}
        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)/100  # word(V) to vector with dim(W)

        # Initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)/np.sqrt(input_dim)    # kaiming init
        self.params["b_proj"] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)/np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)/np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)/np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        # mask : Boolean array of shape (N, T) where mask[i, t] tells whether or not the scores at x[i, t] should contribute to the loss.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        mask = captions_out != self._null   

        # Weight and bias for the affine transform from image features to initial hidden state
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]                                       # Word embedding matrix
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]     # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]      # Weight and bias for the hidden-to-vocab transformation.

        xT, embed_cache = word_embedding_forward(captions_in, W_embed)   # N,T,W
        h0 = features.dot(W_proj) + b_proj   # N, H
        if self.cell_type == 'rnn':
            hT, caches = rnn_forward(xT, h0, Wx, Wh, b)    # hT : N,T,H
        else: 
            hT, caches = lstm_forward(xT, h0, Wx, Wh, b) 
        scores, aff_cache = temporal_affine_forward(hT, W_vocab, b_vocab)   # N,T,V : transforms give hidden states to scores for each word in vocab 
        loss, dout = temporal_softmax_loss(scores, captions_out, mask)   # dout : N,T,V

        grads = {}
        dout, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dout, aff_cache)
        if self.cell_type == 'rnn':
            dxT, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dout, caches)
        else:
            dxT, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dout, caches)
        grads['W_embed'] = word_embedding_backward(dxT, embed_cache)
        grads['W_proj'], grads['b_proj'] = (features.T).dot(dh0), dh0.sum(axis=0)

        # for param in grads.keys():
        #     if param[0] == 'W':
        #         loss += np.sum(self.reg*0.5*self.params[param]**2)
        #         grads[param] += self.reg*self.params[param]       

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        x0 = self._start*np.ones(N, dtype=np.int32)
        h0 = features.dot(W_proj) + b_proj  
        c0 = np.zeros_like(h0) 
        captions[:,0] = x0

        for t in range(1, max_length):
            x0 = W_embed[x0]
            if self.cell_type == 'rnn':
                h0, cache = rnn_step_forward(x0, h0, Wx, Wh, b)
            else:
                h0, c0, cache = lstm_step_forward(x0, h0, c0, Wx, Wh, b)
            scores = h0.dot(W_vocab) + b_vocab
            x0 = scores.argmax(axis=1)
            captions[:, t] = x0

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################

        return captions
