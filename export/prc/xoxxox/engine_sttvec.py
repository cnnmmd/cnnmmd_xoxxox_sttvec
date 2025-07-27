import io
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from xoxxox.shared import Custom
from xoxxox.params import Medium

class SttPrc():

  def __init__(self, config="xoxxox/config_sttvec_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    nmodel = diccnf["nmodel"]
    self.prcstt = Wav2Vec2Processor.from_pretrained(nmodel)
    self.ctcstt = Wav2Vec2ForCTC.from_pretrained(nmodel)

  def infere(self, datwav):
    arrwav, _ = sf.read(io.BytesIO(datwav))
    tsrwav = self.prcstt(arrwav, sampling_rate=Medium.ratsmp, return_tensors="pt", padding=True)
    with torch.no_grad():
      logits = self.ctcstt(tsrwav.input_values, attention_mask=tsrwav.attention_mask).logits
    idsprd = torch.argmax(logits, dim=-1)
    txtprd = self.prcstt.batch_decode(idsprd)
    return txtprd[0]
