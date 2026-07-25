import type { DeviceSpec, ModelSpec, QuantSpec } from '@/engine/types';

/**
 * Whether a quantization is worth offering for a given model and device.
 *
 * One predicate, imported by both the picker and the store. Holding a copy in each is how the
 * control ended up offering options the store immediately replaced — three separate times
 * tonight, in three different rules.
 *
 * Two reasons a format is withheld, and they are different in kind:
 *
 *   - **It cannot run here.** NVFP4 is Blackwell-native, and AMD's published FP4 rate is for its
 *     own format; letting it through hands `peakFlops` a number from different silicon and
 *     produces a plausible, impossible result. This is a hardware fact and the engine also
 *     refuses it in `planPlacement`.
 *   - **It would do nothing.** An expert-only scheme like MXFP4 spares every tensor that is not
 *     a routed expert, so on a dense model it computes exactly BF16 while the label goes on
 *     claiming 4-bit. Not an error — just a no-op with a misleading name, and nothing to learn
 *     from selecting it.
 */
export function quantApplies(quant: QuantSpec, model: ModelSpec, device: DeviceSpec): boolean {
  if (quant.requiresVendor !== undefined && quant.requiresVendor !== device.vendor) return false;
  if (quant.denseBpw !== undefined && model.expertParams === 0) return false;
  return true;
}

/** The safest format that always applies, for when a selection has to be replaced. */
export const FALLBACK_QUANT_ID = 'bf16';
