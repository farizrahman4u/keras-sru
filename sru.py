from keras.layers import Input, Dense, Activation, Lambda, Masking, Dropout, add, multiply, concatenate
from keras.models import Model
from keras import backend as K
from recurrentshop import RecurrentModel, RecurrentSequential


def sru(input, initial_state=None, depth=1, dropout=0.2, recurrent_dropout=0.2, return_sequences=False, **kwargs):
    units = K.int_shape(input)[-1]
    input_masked = Masking(mask_value=0.)(input)
    mask = Lambda(lambda x, mask: mask, output_shape=lambda s: s[:2])(input_masked)
    W = Dense(units * 3)
    def drop(x, p):
        shape = K.shape(x)
        noise_shape = (shape[0], 1, shape[2])
        return Dropout(p, noise_shape=noise_shape).call(x)
    input_dropped = Lambda(drop, arguments={'p': dropout}, output_shape=lambda s: s)(input_masked)
    ones = Lambda(lambda x: x * 0. + 1., output_shape=lambda s: s)(input)
    dropped_ones = Dropout(recurrent_dropout)(ones)
    xfr = W(input_dropped)
    ixfrd = concatenate([input, xfr, dropped_ones])
    ixfrd = Lambda(lambda x: x[0], mask=lambda x, _: x[1])([ixfrd, mask])
    recurrent_input = Input((units * 5,))
    def unpack(x, n):
        return [Lambda(lambda x, i: x[:,units * i : units * (i + 1)], arguments={'i': i}, output_shape=lambda s: (s[0], units))(x) for i in range(n)]
    x_t, x_p_t, f_t, r_t, drop = unpack(recurrent_input, 5)
    f_t = Activation('sigmoid')(f_t)
    r_t = Activation('sigmoid')(r_t)
    inv = Lambda(lambda x: 1. - x, output_shape=lambda s: s)
    c_tm1 = Input((units, ))
    c_t = c_tm1
    h_t = x_t
    for _ in range(depth):
    	c_t = add([multiply([f_t, c_t]), multiply([inv(f_t), x_p_t])])
    	c_t = multiply([c_t, drop])
    	h_t = add([multiply([r_t, Activation('tanh')(c_t)]), multiply([inv(r_t), h_t])])
    	xfr = W(h_t)
    	x_p_t, f_t, r_t = unpack(xfr, 3)
    rnn = RecurrentModel(recurrent_input, h_t, c_tm1, c_t, return_sequences=return_sequences, **kwargs)
    output = rnn(ixfrd, initial_state=initial_state)
    return output
