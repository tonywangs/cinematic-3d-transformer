"""Read-only hooks on native GPT-NeoX; reproducible, full-precision trace shards."""
import hashlib,json,time,random
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
MODEL='EleutherAI/pythia-70m';REVISION='de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852'
PROMPTS=[('story','A beginning','Once upon a time, there was a'),('pattern','A pattern',' red blue green red blue green red'),('rhyme','A rhyme','The sun is bright, the moon at night,'),('list','A list','1. Apples\n2. Oranges\n3.'),('code','A function','def square(x):\n    return'),('ambiguous','An open door','She opened the door and saw')]
root=Path(__file__).resolve().parents[1];out=root/'public/data';out.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32,attn_implementation='eager').eval();cfg=model.config
assert (cfg.num_hidden_layers,cfg.num_attention_heads,cfg.hidden_size,cfg.intermediate_size,cfg.vocab_size,cfg.use_parallel_residual)==(6,8,512,2048,50304,True)
checks=[]
def trace(ids):
 with torch.inference_mode():ref=model(ids,use_cache=False).logits
 data={};handles=[]
 def save(name,value):data[name]=value.detach().cpu().contiguous().clone()
 def hook(module,name,pre=False):
  if pre: handles.append(module.register_forward_pre_hook(lambda m,args:save(name,args[0])))
  else:handles.append(module.register_forward_hook(lambda m,args,result:save(name,result)))
 hook(model.gpt_neox.embed_in,'embedding');hook(model.gpt_neox.final_layer_norm,'final_norm')
 for l,block in enumerate(model.gpt_neox.layers):
  p=f'layer.{l}.';hook(block,p+'residual_in',True);hook(block,p+'residual_out');hook(block.input_layernorm,p+'attention_norm');hook(block.post_attention_layernorm,p+'mlp_norm');hook(block.mlp.dense_h_to_4h,p+'mlp_pre');hook(block.mlp.dense_4h_to_h,p+'mlp_post',True);hook(block.mlp,p+'mlp_out');hook(block.attention.query_key_value,p+'qkv');hook(block.attention.dense,p+'heads',True);hook(block.attention.dense,p+'attention_out')
  def attention(m,args,kwargs,result,p=p):
   q,k,v=data[p+'qkv'].view(1,ids.shape[1],8,192).transpose(1,2).chunk(3,-1);save(p+'q_before_rope',q);save(p+'k_before_rope',k);save(p+'v',v);q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings']);save(p+'q',q);save(p+'k',k);save(p+'attention',result[1]);reconstructed=(result[1]@v).transpose(1,2).reshape(1,ids.shape[1],512);assert torch.allclose(reconstructed,data[p+'heads'],atol=1e-6,rtol=1e-6)
  handles.append(block.attention.register_forward_hook(attention,with_kwargs=True))
 try:
  with torch.inference_mode():actual=model(ids,use_cache=False).logits
 finally:
  for h in handles:h.remove()
 error=float((ref-actual).abs().max());assert error<1e-4;assert torch.equal(ref.argmax(-1),actual.argmax(-1));save('logits',actual[0,-1])
 for l in range(6):
  p=f'layer.{l}.';a=data[p+'attention'];assert float((a.sum(-1)-1).abs().max())<1e-5;assert float(a.triu(1).abs().max())==0
  assert torch.equal(data[p+'mlp_out']+data[p+'attention_out']+data[p+'residual_in'],data[p+'residual_out'])
  del data[p+'qkv']
 prob=actual[0,-1].double().softmax(-1);assert abs(float(prob.sum())-1)<1e-5,(float(prob.sum()),float(actual.min()),float(actual.max()))
 return data,prob,error
catalog=[]
for slug,title,prompt in PROMPTS:
 ids=tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids;assert ids.shape[1]<=48;initial=ids[0].tolist();steps=[];directory=out/slug;directory.mkdir(exist_ok=True)
 for step in range(8):
  data,prob,error=trace(ids);buffer=bytearray();tensors={}
  for name,value in data.items():
   array=value.numpy().astype('<f4');raw=array.tobytes();tensors[name]={'shape':list(array.shape),'offset':len(buffer),'length':array.size,'samples':[{'index':int(i),'value':float(array.flat[i])} for i in np.linspace(0,array.size-1,min(10,array.size),dtype=int)]};buffer.extend(raw)
  file=f'step-{step}.bin';(directory/file).write_bytes(buffer);p,ix=prob.topk(10);nextid=int(ix[0]);steps.append({'file':f'/data/{slug}/{file}','bytes':len(buffer),'sha256':hashlib.sha256(buffer).hexdigest(),'tensors':tensors,'token_ids':ids[0].tolist(),'tokens':[tok.decode([i]) for i in ids[0]],'next_id':nextid,'next_token':tok.decode([nextid]),'predictions':[{'id':int(i),'text':tok.decode([i]),'p':float(v)} for i,v in zip(ix,p)],'remaining_mass':1-float(p.sum()),'parity_max_abs':error});ids=torch.cat([ids,torch.tensor([[nextid]])],1);print(slug,step,repr(tok.decode([nextid])),len(buffer),flush=True)
 assert sum(s['bytes'] for s in steps)<50_000_000
 manifest={'version':'TraceBundle/1','model':MODEL,'revision':REVISION,'tokenizer_revision':REVISION,'architecture':{'layers':6,'heads':8,'hidden':512,'mlp':2048,'vocabulary':50304,'parallel_residual':True,'rope_dimensions_per_head':model.gpt_neox.layers[0].attention.rotary_ndims},'dtype':'float32 little-endian','torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','attention':'eager','kv_cache':False,'prompt':prompt,'initial_ids':initial,'generated_ids':ids[0,len(initial):].tolist(),'continuation':tok.decode(ids[0,len(initial):]),'extraction_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':steps}
 (directory/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')));catalog.append({'id':slug,'title':title,'prompt':prompt,'manifest':f'/data/{slug}/manifest.json'});checks.append({'prompt':prompt,'steps':8,'max_abs':max(s['parity_max_abs'] for s in steps),'bytes':sum(s['bytes'] for s in steps)})
random.seed(19)
words=['tree','river','blue','apple','one','cat','because','cloud','light','runs','a','the']
for i in range(20):
 prompt=' '.join(random.choices(words,k=random.randint(3,18)));_,_,error=trace(tok(prompt,return_tensors='pt').input_ids);checks.append({'seeded_prompt':prompt,'max_abs':error})
(out/'gallery.json').write_text(json.dumps(catalog));(root/'docs/validation.json').write_text(json.dumps({'seed':19,'checks':checks,'checks_per_trace':['native logits parity <1e-4','same greedy token','causal zeros','attention rows sum to one','weighted value reconstruction','parallel residual exact recombination','softmax normalization','ten source samples per tensor']},indent=2))
