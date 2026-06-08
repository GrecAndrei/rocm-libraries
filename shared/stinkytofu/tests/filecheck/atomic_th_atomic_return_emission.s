# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm
#
# Return-form flat_atomic must round-trip with th:TH_ATOMIC_RETURN
# (assembler rejects without it).

# CHECK: flat_atomic_dec_u32 v0, v[1:2], v3 th:TH_ATOMIC_RETURN
# CHECK: flat_atomic_cmpswap_b32 v[1:2], v[3:4] th:TH_ATOMIC_RETURN

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

flat_atomic_dec_u32 v0, v[1:2], v3 th:TH_ATOMIC_RETURN
flat_atomic_cmpswap_b32 v[1:2], v[3:4]
s_endpgm
